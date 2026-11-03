"""Stage 6: probabilities -> step events, and scoring against labels."""

import numpy as np

from laptop.vision.compare import compare
from laptop.vision.steps import LEGS


def pick_peaks(prob, rate_hz, threshold=0.5, refractory_s=0.3):
    """Sample indices of local maxima above threshold, >= refractory apart (biggest wins)."""
    p = np.asarray(prob, dtype=float)
    if len(p) < 3:
        return []
    cand = np.flatnonzero((p[1:-1] >= threshold) & (p[1:-1] >= p[:-2]) & (p[1:-1] > p[2:])) + 1
    gap = refractory_s * rate_hz
    keep = []
    for i in sorted(cand, key=lambda i: -p[i]):
        if all(abs(i - j) >= gap for j in keep):
            keep.append(int(i))
    return sorted(keep)


def predict_events(probs, rate_hz, threshold, refractory_s):
    """probs (T, 2) -> [(sample_idx, leg)]."""
    return sorted((i, leg) for k, leg in enumerate(LEGS)
                  for i in pick_peaks(probs[:, k], rate_hz, threshold, refractory_s))


def score(pred, truth, labeled, rate_hz, tol_s):
    """Compare events inside the labeled region. Returns compare() dict + summary."""
    inside = lambda ev: [(i, leg) for i, leg in ev if labeled[min(i, len(labeled) - 1)]]
    pred, truth = inside(pred), inside(truth)
    tol = int(round(tol_s * rate_hz))
    r = compare(pred, truth, tol)
    blind = compare([(i, "L") for i, _ in pred], [(i, "L") for i, _ in truth], tol)["total"]
    tot = r["total"]
    prec = tot["matched"] / tot["detected"] if tot["detected"] else 0.0
    rec = tot["matched"] / tot["truth"] if tot["truth"] else 0.0
    errs = [e for leg in LEGS for e in r[leg]["errors"]]
    return {
        "per_leg": {leg: {k: r[leg][k] for k in ("detected", "truth", "matched")} for leg in LEGS},
        "truth": tot["truth"], "predicted": tot["detected"], "matched": tot["matched"],
        "precision": prec, "recall": rec,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "count_error": tot["count_error"],
        "timing_median_abs_ms": float(np.median(np.abs(errs)) / rate_hz * 1000) if errs else None,
        "timing_mean_ms": float(np.mean(errs) / rate_hz * 1000) if errs else None,
        "leg_blind_matched": blind["matched"],
    }


class BoutGate:
    """Count steps only inside walking bouts (what phone pedometers do).

    A step that follows the previous counted step within max_gap_s is counted
    at once. Otherwise steps wait as "pending" until min_steps of them arrive,
    each within max_gap_s of the last (and, if need_both_legs, from both
    legs) -- then they are all counted together, retroactively. An isolated
    ankle twitch or tapping one foot never forms a bout. min_steps <= 1
    turns the gate off.
    """

    def __init__(self, min_steps=4, max_gap_s=1.5, need_both_legs=True):
        self.min_steps, self.max_gap_s, self.need_both_legs = min_steps, max_gap_s, need_both_legs
        self.reset()

    def reset(self):
        self.pending = []      # [(t_ns, leg, item)]
        self._last_t = None    # last counted step

    @property
    def walking(self):
        return self._last_t is not None

    def push(self, t_ns, leg, item):
        """Returns the items to count now (maybe several, maybe none)."""
        gap = self.max_gap_s * 1e9
        if self.min_steps <= 1 or (self._last_t is not None and t_ns - self._last_t <= gap):
            self._last_t = t_ns
            return [item]
        self._last_t = None
        if self.pending and t_ns - self.pending[-1][0] > gap:
            self.pending = []
        self.pending = (self.pending + [(t_ns, leg, item)])[-3 * self.min_steps:]
        legs = {l for _, l, _ in self.pending}
        if len(self.pending) >= self.min_steps and (not self.need_both_legs or len(legs) == 2):
            out = [it for _, _, it in self.pending]
            self.pending = []
            self._last_t = t_ns
            return out
        return []


def bout_filter(events, t_ns, min_steps=4, max_gap_s=1.5, need_both_legs=True):
    """Offline version: [(sample_idx, leg)] -> the ones inside walking bouts."""
    gate = BoutGate(min_steps, max_gap_s, need_both_legs)
    out = []
    for i, leg in sorted(events):
        out += gate.push(int(t_ns[i]), leg, (i, leg))
    return out
