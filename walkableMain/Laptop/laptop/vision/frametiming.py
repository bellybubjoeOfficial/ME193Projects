"""Frame-timestamp analysis: measured fps, dropped frames, fps stability.

Pure functions on a list of per-frame laptop timestamps (ns), shared by the
live recorder and the Stage 2 pass test.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Drop:
    after_frame: int   # last good frame before the hole
    t_ns: int          # its timestamp
    missing: int       # frames that should have arrived in the hole
    interval_ms: float


def _long_intervals(t, period, drop_factor):
    """Split long intervals into real holes and late timestamps.

    A frame stamped late (the stamping thread was briefly blocked) makes a
    long interval followed by a short one that cancels it. Measuring across
    two intervals tells them apart: a real hole still spans an extra period.
    """
    dt = np.diff(t)
    holes, late = [], []
    for i in np.flatnonzero(dt > drop_factor * period):
        if i + 2 < len(t):
            missing = int(round((t[i + 2] - t[i]) / period)) - 2
        else:
            missing = int(round(dt[i] / period)) - 1
        if missing >= 1:
            holes.append((int(i), missing))
        else:
            late.append((int(i), dt[i] - period))
    return holes, late


def find_drops(t_ns, nominal_fps, drop_factor=1.5):
    """Holes where frames were really lost (not just stamped late)."""
    t = np.asarray(t_ns, dtype=np.int64)
    if len(t) < 2:
        return []
    period = 1e9 / nominal_fps
    holes, _ = _long_intervals(t, period, drop_factor)
    return [Drop(i, int(t[i]), m, (t[i + 1] - t[i]) / 1e6) for i, m in holes]


def find_late_stamps(t_ns, nominal_fps, drop_factor=1.5):
    """[(frame_idx, lateness_ms)] for frames whose timestamp was taken late."""
    t = np.asarray(t_ns, dtype=np.int64)
    if len(t) < 3:
        return []
    _, late = _long_intervals(t, 1e9 / nominal_fps, drop_factor)
    return [(i + 1, lag / 1e6) for i, lag in late]


def mean_fps(t_ns):
    t = np.asarray(t_ns, dtype=np.int64)
    if len(t) < 2 or t[-1] == t[0]:
        return 0.0
    return (len(t) - 1) / ((t[-1] - t[0]) / 1e9)


def window_fps(t_ns, window_s):
    """fps in consecutive windows (frames counted per window / window length).

    The last partial window is dropped.
    """
    t = np.asarray(t_ns, dtype=np.int64)
    if len(t) < 2:
        return np.array([])
    rel = (t - t[0]) / 1e9
    n_windows = int(rel[-1] // window_s)
    edges = np.arange(n_windows + 1) * window_s
    counts = np.histogram(rel, bins=edges)[0] if n_windows else np.array([])
    return counts / window_s


def fps_stability(t_ns, window_s, tol_frac):
    """Worst window's deviation from the run mean, and whether it's within tol."""
    fps = window_fps(t_ns, window_s)
    m = mean_fps(t_ns)
    if not len(fps) or m == 0:
        return {"mean_fps": m, "window_fps": fps, "worst_dev_frac": None, "stable": False}
    worst = float(np.max(np.abs(fps / m - 1)))
    return {"mean_fps": m, "window_fps": fps, "worst_dev_frac": worst, "stable": worst <= tol_frac}
