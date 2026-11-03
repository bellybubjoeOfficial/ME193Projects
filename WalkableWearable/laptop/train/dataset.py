"""Stage 6: turn a synced session into model inputs and targets.

IMU samples go on a uniform grid indexed by seq (a lost sample leaves a
zero row with valid=False). Targets: one channel per leg, a Gaussian bump
(label_sigma_s) at each heel strike. `labeled` marks where the targets can
be trusted: between the stomps (minus a margin) and where the camera could
see a heel. Loss and evaluation only count labeled samples.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from laptop.data.load import load_imu
from laptop.vision.steps import LEGS

CHANNELS = ["ax", "ay", "az", "gx", "gy", "gz"]


@dataclass
class SessionData:
    name: str
    t_ns: np.ndarray       # (n,) int64 IMU laptop time per grid sample
    x: np.ndarray          # (n, 6) float32 raw units, zeros where missing
    valid: np.ndarray      # (n,) bool  IMU sample present
    labeled: np.ndarray    # (n,) bool  targets trustworthy here
    y: np.ndarray          # (n, 2) float32 soft targets, columns = LEGS
    events: list = field(default_factory=list)   # [(sample_idx, leg)]
    meta: dict = field(default_factory=dict)
    rate_hz: float = 100.0


def imu_grid(imu):
    """Seq-indexed uniform grid from a load_imu() frame (sorted by seq)."""
    imu = imu.dropna(subset=["t_laptop_ns"])
    seq = imu["seq"].to_numpy(dtype=np.int64)
    idx = seq - seq[0]
    n = int(idx[-1]) + 1
    x = np.zeros((n, len(CHANNELS)), dtype=np.float32)
    x[idx] = imu[CHANNELS].to_numpy(dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    valid[idx] = True
    t = np.interp(np.arange(n), idx, imu["t_laptop_ns"].to_numpy(dtype=np.float64)).astype(np.int64)
    return t, x, valid


def soft_targets(t_ns, events_t, events_leg, sigma_s):
    y = np.zeros((len(t_ns), len(LEGS)), dtype=np.float32)
    sig = sigma_s * 1e9
    for te, leg in zip(events_t, events_leg):
        k = LEGS.index(leg)
        m = np.abs(t_ns - te) < 4 * sig
        y[m, k] = np.maximum(y[m, k], np.exp(-0.5 * ((t_ns[m] - te) / sig) ** 2))
    return y


def labeled_mask(t_ns, frame_t, seen, stomp_ts, margin_s):
    """Where targets can be trusted (see module doc)."""
    vis = np.interp(t_ns, frame_t, seen.astype(float), left=0, right=0) > 0.5
    m = vis.copy()
    if len(stomp_ts) == 2:
        m &= (t_ns > stomp_ts[0] + margin_s * 1e9) & (t_ns < stomp_ts[1] - margin_s * 1e9)
    for ts in stomp_ts:
        m &= np.abs(t_ns - ts) > margin_s * 1e9
    return m


def load_session(folder, cfg, relabel=False, verbose=True):
    from laptop.train.labels import session_labels
    folder = Path(folder)
    tc = cfg["train"]
    t, x, valid = imu_grid(load_imu(folder))
    labels, frame_t, seen = session_labels(folder, cfg, relabel, verbose)
    stomps = []
    if (folder / "stomp.json").exists():
        st = json.load(open(folder / "stomp.json"))["stomps"]
        stomps = [st[k]["imu"]["t_ns"] for k in ("start", "end") if st[k]["imu"]]
    labeled = labeled_mask(t, frame_t, seen, stomps, tc["stomp_margin_s"]) & valid
    ev_t, ev_leg = labels["t_imu_ns"].to_numpy(), labels["leg"].to_numpy()
    y = soft_targets(t, ev_t, ev_leg, tc["label_sigma_s"])
    events = [(int(np.argmin(np.abs(t - te))), leg) for te, leg in zip(ev_t, ev_leg)]
    meta = json.load(open(folder / "meta.json")) if (folder / "meta.json").exists() else {}
    rate = (len(t) - 1) / ((t[-1] - t[0]) / 1e9)
    return SessionData(folder.name, t, x, valid, labeled, y, events, meta, rate)


class Normalizer:
    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    @classmethod
    def fit(cls, sessions):
        xs = np.concatenate([s.x[s.valid] for s in sessions])
        return cls(xs.mean(0), xs.std(0) + 1e-6)

    def __call__(self, x):
        return (x - self.mean) / self.std

    def to_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


def sample_windows(sessions, norm, n, length, min_labeled, rng, gain=0.0, noise=0.0):
    """Random training windows: (x (n, 6, L), y (n, 2, L), mask (n, L))."""
    weights = np.array([max(s.labeled.sum(), 0) for s in sessions], dtype=float)
    weights /= weights.sum()
    X = np.zeros((n, len(CHANNELS), length), dtype=np.float32)
    Y = np.zeros((n, len(LEGS), length), dtype=np.float32)
    M = np.zeros((n, length), dtype=np.float32)
    for i in range(n):
        for _ in range(50):  # find a window that is mostly labeled
            s = sessions[rng.choice(len(sessions), p=weights)]
            a = rng.integers(0, len(s.t_ns) - length)
            if s.labeled[a:a + length].mean() >= min_labeled:
                break
        xw = norm(s.x[a:a + length]) * s.valid[a:a + length, None]
        if gain:
            xw = xw * (1 + rng.normal(0, gain, (1, len(CHANNELS))))
        if noise:
            xw = xw + rng.normal(0, noise, xw.shape)
        X[i] = xw.T
        Y[i] = s.y[a:a + length].T
        M[i] = s.labeled[a:a + length]
    return X, Y, M
