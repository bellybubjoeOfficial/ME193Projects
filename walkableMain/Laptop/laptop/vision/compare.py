"""Compare detected heel strikes with hand-marked truth (per leg).

Matching is one-to-one: closest pairs first, only within tol frames.
Timing error = detected frame - true frame (positive = detector late).
"""

import numpy as np

from laptop.vision.steps import LEGS


def match_events(det, truth, tol):
    """det, truth: frame indices. Returns (pairs [(d, t)], missed [t], extra [d])."""
    cands = sorted((abs(d - t), i, j) for i, d in enumerate(det) for j, t in enumerate(truth)
                   if abs(d - t) <= tol)
    used_d, used_t, pairs = set(), set(), []
    for _, i, j in cands:
        if i not in used_d and j not in used_t:
            used_d.add(i)
            used_t.add(j)
            pairs.append((det[i], truth[j]))
    pairs.sort(key=lambda p: p[1])
    missed = [t for j, t in enumerate(truth) if j not in used_t]
    extra = [d for i, d in enumerate(det) if i not in used_d]
    return pairs, missed, extra


def compare(det_events, truth_events, tol):
    """Both: iterables of (frame_idx, leg). Returns per-leg and total results."""
    out = {}
    all_err = []
    for leg in LEGS:
        d = sorted(f for f, l in det_events if l == leg)
        t = sorted(f for f, l in truth_events if l == leg)
        pairs, missed, extra = match_events(d, t, tol)
        err = [a - b for a, b in pairs]
        all_err += err
        out[leg] = {"detected": len(d), "truth": len(t), "matched": len(pairs),
                    "missed": missed, "extra": extra, "errors": err}
    n_det = sum(out[l]["detected"] for l in LEGS)
    n_truth = sum(out[l]["truth"] for l in LEGS)
    abs_err = np.abs(all_err)
    out["total"] = {
        "detected": n_det, "truth": n_truth, "count_error": n_det - n_truth,
        "matched": sum(out[l]["matched"] for l in LEGS),
        "missed": sum(len(out[l]["missed"]) for l in LEGS),
        "extra": sum(len(out[l]["extra"]) for l in LEGS),
        "median_abs_err": float(np.median(abs_err)) if len(abs_err) else None,
        "mean_err": float(np.mean(all_err)) if all_err else None,
        "max_abs_err": int(abs_err.max()) if len(abs_err) else None,
    }
    return out


def format_report(name, r):
    tot = r["total"]
    lines = [f"{name}: detected {tot['detected']}  true {tot['truth']}  "
             f"(count error {tot['count_error']:+d})"]
    for leg in LEGS:
        x = r[leg]
        lines.append(f"  {leg}: detected {x['detected']:3d}  true {x['truth']:3d}  matched {x['matched']:3d}"
                     f"  missed {x['missed'] or '-'}  extra {x['extra'] or '-'}")
    if tot["median_abs_err"] is not None:
        lines.append(f"  timing: median |error| {tot['median_abs_err']:.1f} frames, "
                     f"mean {tot['mean_err']:+.2f} (+ = detector late), max {tot['max_abs_err']}")
    return "\n".join(lines)
