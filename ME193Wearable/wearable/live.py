"""Stages 6-7: live inference on a stream of samples (runs inside the App, numpy only).

    p = LivePredictor(forest_lite.load("models/model2.json"))
    for each sample row:  out = p.add(row)       # None, or a prediction dict every 1.28 s

How it matches training:
  * training filtered gravity over a WHOLE recording; live keeps a rolling
    buffer (LIVE_BUFFER_SECONDS) and filters that, then takes the newest
    256-sample window from it. tests/test_stage6_model2.py checks that the two
    give (nearly) the same predictions.
  * a new window every 128 samples = 50 % overlap, same as training.
  * a majority vote over the last LIVE_VOTE_WINDOWS predictions smooths out
    single wrong windows (at the cost of ~2 s extra delay when you switch).

Model 1 and Model 2 use the same code: each model file lists the features it needs.
"""

from collections import Counter, deque

import numpy as np

from wearable import config, features, forest_lite, imu

ACC_IDX = [imu.CSV_COLUMNS.index(c) for c in imu.ACCEL_COLS]
GYR_IDX = [imu.CSV_COLUMNS.index(c) for c in imu.GYRO_COLS]


class LivePredictor:
    def __init__(self, model, fs=config.SAMPLE_RATE_HZ, window=config.WINDOW_SAMPLES,
                 overlap=config.WINDOW_OVERLAP, buffer_s=config.LIVE_BUFFER_SECONDS,
                 vote=config.LIVE_VOTE_WINDOWS):
        self.model = model
        self.fs, self.window = fs, window
        self.step = max(1, int(round(window * (1 - overlap))))
        self.buf = deque(maxlen=max(window, int(buffer_s * fs)))
        self.votes = deque(maxlen=vote)
        self.since_last = 0
        self.n_predictions = 0

    def add(self, row):
        """Add one sample (a row in imu.CSV_COLUMNS order). Returns a prediction or None."""
        self.buf.append([row[i] for i in ACC_IDX] + [row[i] for i in GYR_IDX])
        self.since_last += 1
        if len(self.buf) < self.window or self.since_last < self.step:
            return None
        self.since_last = 0
        return self.predict_now()

    def predict_now(self):
        data = np.asarray(self.buf, dtype=float)
        acc, gyr = data[:, :3], data[:, 3:]
        grav = features.gravity_lowpass(acc, self.fs, use_scipy=False)
        w = slice(len(data) - self.window, len(data))
        feats = features.window_features(acc[w], gyr[w], grav[w], self.fs)
        proba = forest_lite.predict_proba(self.model, forest_lite.features_from_row(self.model, feats))[0]
        raw = self.model["classes"][int(np.argmax(proba))]
        self.votes.append(raw)
        label, n = Counter(self.votes).most_common(1)[0]
        self.n_predictions += 1
        return {
            "label": label,                         # smoothed (majority vote): show this
            "raw_label": raw,                       # this window alone
            "vote_fraction": n / len(self.votes),
            "proba": {c: round(float(p), 3) for c, p in zip(self.model["classes"], proba)},
            "n": self.n_predictions,
        }


def replay(model, df):
    """Feed a recorded DataFrame through the live predictor; list of predictions."""
    p = LivePredictor(model)
    out = []
    for row in df[imu.CSV_COLUMNS].itertuples(index=False):
        r = p.add(row)
        if r is not None:
            out.append(r)
    return out
