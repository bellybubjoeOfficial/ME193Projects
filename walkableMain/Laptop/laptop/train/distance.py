"""Stage 8: distance = sum of step lengths, three ways.

Step-length methods (each fitted on some sessions, tested on others):
  prior     your Apple step length (0.63 m) for every step -- no fitting
  fixed     one calibrated step length
  cadence   step = a + b * step_frequency  (1 / time since the previous step)
  pendulum  the IMU shin's swing angle theta over each of its strides;
            stride = k * 2 * leg * sin(theta / 2), each step = stride / 2
  walk_ratio      step = r * cadence (steps/min). The "walk ratio" r is nearly
                  constant across walking speeds for one person, so faster
                  cadence -> longer steps. r fitted ...
  walk_ratio_prior  ... or straight from Apple (0.63 m / 114 steps/min)
  swing_speed     step = a + b * peak shin rotation speed in the stride (deg/s),
                  which rises with walking speed

Ground truth comes from the video, with YOUR LEG as the ruler: a straight
leg's hip-to-ankle distance in pixels = leg_length - ankle_height in metres.
(The floor tape was tried first; it lies nearer the camera than your feet,
so it overstated the scale by ~30%.) A "pass" is a straight stretch where
the hips keep moving the same way; its true length is the hip travel.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

METHODS = ["prior", "fixed", "cadence", "pendulum", "walk_ratio_prior", "walk_ratio", "swing_speed"]


# ---------------- ground truth from video ----------------
def px_per_m(lm, cfg, vis=0.7):
    """Pixels per metre at the walker's depth, from the straight-leg length."""
    d = cfg["distance"]
    lengths = []
    for s in ("left", "right"):
        ok = (lm[f"{s}_hip_vis"] > vis) & (lm[f"{s}_ankle_vis"] > vis)
        lengths.append(np.hypot(lm[f"{s}_hip_x"] - lm[f"{s}_ankle_x"],
                                lm[f"{s}_hip_y"] - lm[f"{s}_ankle_y"])[ok])
    lengths = pd.concat(lengths)
    if len(lengths) < 30:
        return None
    # A straight leg is the long end of the distribution (knees are bent most of the time).
    return float(np.percentile(lengths, 90)) / (d["leg_length_m"] - d["ankle_height_m"])


def straight_passes(lm, cfg):
    """[(t_start_ns, t_end_ns, travel_px)] straight walking stretches (video clock)."""
    d = cfg["distance"]
    t = lm["t_ns"].to_numpy(dtype=np.int64)
    fps = (len(t) - 1) / ((t[-1] - t[0]) / 1e9)
    hip = ((lm["left_hip_x"] + lm["right_hip_x"]) / 2).interpolate(limit=5, limit_area="inside")
    hs = hip.rolling(9, center=True, min_periods=5).mean().to_numpy()
    v = np.r_[np.nan, np.diff(hs)] * fps
    speed = np.abs(v)
    if not np.isfinite(speed).any():
        return []
    walking = speed > d["walking_speed_frac"] * np.nanpercentile(speed, 75)
    direction = np.sign(np.nan_to_num(v))
    out, a = [], None
    for i in range(len(t) + 1):
        ok = i < len(t) and walking[i] and np.isfinite(hs[i])
        if ok and a is not None and direction[i] == direction[a]:
            continue
        if a is not None and (t[i - 1] - t[a]) / 1e9 >= d["min_straight_s"]:
            out.append((int(t[a]), int(t[i - 1]), float(abs(hs[i - 1] - hs[a]))))
        a = i if ok else None
    return out


# ---------------- per-step IMU features ----------------
def sagittal_axis(gyro):
    """Unit vector of the gyro's dominant rotation axis (the shin swinging fore-aft)."""
    g = np.asarray(gyro, dtype=float)
    g = g[np.isfinite(g).all(1)]
    w, vecs = np.linalg.eigh(np.cov(g.T))
    return vecs[:, np.argmax(w)]


def swing_angle_deg(gyro_seg, rate_hz, axis):
    """Peak-to-peak shin angle over one stride: integrate the sagittal rate, minus drift."""
    w = np.asarray(gyro_seg, dtype=float) @ axis
    ang = np.cumsum(w) / rate_hz
    ang -= np.linspace(ang[0], ang[-1], len(ang))  # bias -> linear drift over the stride
    return float(np.ptp(ang))


def swing_peak_dps(gyro_seg, axis):
    """Peak shin rotation speed in the stride (deg/s, either direction)."""
    return float(np.max(np.abs(np.asarray(gyro_seg, dtype=float) @ axis)))


def stride_from_angle(theta_deg, leg_m):
    return 2 * leg_m * np.sin(np.radians(theta_deg) / 2)


def step_features(step_t, step_leg, imu_t, gyro, rate_hz, imu_leg, axis, cfg):
    """DataFrame per step: t_ns, leg, freq (Hz, 1/gap), pend (m, pendulum step length).

    Missing values (first step of a bout, strides out of range) are filled with
    the session median so every step has a length.
    """
    d = cfg["distance"]
    order = np.argsort(step_t)
    t = np.asarray(step_t, dtype=np.int64)[order]
    legs = np.asarray(step_leg)[order]
    gap = np.r_[np.nan, np.diff(t) / 1e9]
    freq = np.where((gap >= d["step_gap_s"][0]) & (gap <= d["step_gap_s"][1]), 1 / gap, np.nan)
    pend = np.full(len(t), np.nan)
    gpk = np.full(len(t), np.nan)
    mine = np.flatnonzero(legs == imu_leg)
    for a, b in zip(mine, mine[1:]):
        dur = (t[b] - t[a]) / 1e9
        if not d["stride_s"][0] <= dur <= d["stride_s"][1]:
            continue
        seg = (imu_t >= t[a]) & (imu_t < t[b])
        if seg.sum() < 0.5 * dur * rate_hz:
            continue
        theta = swing_angle_deg(gyro[seg], rate_hz, axis)
        pend[a + 1:b + 1] = stride_from_angle(theta, d["leg_length_m"]) / 2
        gpk[a + 1:b + 1] = swing_peak_dps(gyro[seg], axis)
    df = pd.DataFrame({"t_ns": t, "leg": legs, "freq": freq, "pend": pend, "gpk": gpk})
    for c, default in (("freq", 1.9), ("pend", 0.6), ("gpk", 250.0)):
        med = df[c].median()
        df[c] = df[c].fillna(med if np.isfinite(med) else default)
    return df


# ---------------- models ----------------
def lengths(method, params, feats, cfg):
    if method == "prior":
        return np.full(len(feats), cfg["distance"]["prior_step_m"])
    if method == "fixed":
        return np.full(len(feats), params["L"])
    if method == "cadence":
        return params["a"] + params["b"] * feats["freq"].to_numpy()
    if method == "pendulum":
        return params["k"] * feats["pend"].to_numpy()
    if method in ("walk_ratio", "walk_ratio_prior"):
        r = params.get("r") or cfg["distance"]["prior_step_m"] / cfg["distance"]["prior_cadence_spm"]
        return r * 60 * feats["freq"].to_numpy()
    if method == "swing_speed":
        return params["a"] + params["b"] * feats["gpk"].to_numpy()
    raise ValueError(method)


def fit(method, passes):
    """passes: [(feats DataFrame of the steps in the pass, true_m)] -> params (least squares on totals)."""
    true = np.array([m for _, m in passes])
    n = np.array([len(f) for f, _ in passes], dtype=float)
    if method in ("prior", "walk_ratio_prior"):
        return {}
    if method == "fixed":
        return {"L": float(true.sum() / n.sum())}
    if method == "cadence":
        X = np.column_stack([n, [f["freq"].sum() for f, _ in passes]])
        (a, b), *_ = np.linalg.lstsq(X, true, rcond=None)
        return {"a": float(a), "b": float(b)}
    if method == "pendulum":
        S = np.array([f["pend"].sum() for f, _ in passes])
        return {"k": float((S @ true) / (S @ S))}
    if method == "walk_ratio":
        C = np.array([60 * f["freq"].sum() for f, _ in passes])
        return {"r": float((C @ true) / (C @ C))}
    if method == "swing_speed":
        X = np.column_stack([n, [f["gpk"].sum() for f, _ in passes]])
        (a, b), *_ = np.linalg.lstsq(X, true, rcond=None)
        return {"a": float(a), "b": float(b)}
    raise ValueError(method)


@dataclass
class DistanceTracker:
    """Live running distance: add each counted step, read total_m and band_m."""
    method: str
    params: dict
    cfg: dict
    err_frac: float = 0.1      # held-out error of this method (for the +- band)
    axis: np.ndarray = None
    imu_leg: str = "R"
    total_m: float = 0.0
    steps: int = 0
    _last_t: int = None
    _last_mine: int = None
    _pend: float = None
    _gpk: float = None

    def add(self, t_ns, leg, gyro_since_last_mine=None, rate_hz=100.0):
        """gyro_since_last_mine: gyro samples since the previous IMU-leg step (pendulum only)."""
        d = self.cfg["distance"]
        gap = None if self._last_t is None else (t_ns - self._last_t) / 1e9
        freq = 1 / gap if gap and d["step_gap_s"][0] <= gap <= d["step_gap_s"][1] else 1.9
        if leg == self.imu_leg:
            if (self._last_mine is not None and gyro_since_last_mine is not None
                    and d["stride_s"][0] <= (t_ns - self._last_mine) / 1e9 <= d["stride_s"][1]
                    and self.axis is not None and len(gyro_since_last_mine) > 10):
                theta = swing_angle_deg(gyro_since_last_mine, rate_hz, self.axis)
                self._pend = stride_from_angle(theta, d["leg_length_m"]) / 2
                self._gpk = swing_peak_dps(gyro_since_last_mine, self.axis)
            self._last_mine = t_ns
        feats = pd.DataFrame({"freq": [freq], "pend": [self._pend if self._pend else 0.6],
                              "gpk": [self._gpk if self._gpk else self.params.get("gpk_default", 250.0)]})
        step = float(lengths(self.method, self.params, feats, self.cfg)[0])
        self.total_m += step
        self.steps += 1
        self._last_t = t_ns
        return step

    @property
    def band_m(self):
        return self.total_m * self.err_frac

    def reset(self):
        self.total_m, self.steps = 0.0, 0
        self._last_t = self._last_mine = self._pend = self._gpk = None
