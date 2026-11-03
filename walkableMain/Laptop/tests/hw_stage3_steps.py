"""
Stage 3 hardware test -- vision-only step detector vs your hand-marked truth.

For each clip: runs pose (cached) + the detector, compares with truth.csv
from the hand-check tool, and explains every miss/extra using landmark
visibility. Exits 0 if every clip passes.

Workflow (from ME193WalkWearable/), camera side-on, ~20 steps per clip:
    python -m laptop.vision.capture --label slow_walk  --minutes 0.5
    python -m laptop.vision.capture --label normal_walk --minutes 0.5
    python -m laptop.vision.capture --label jog        --minutes 0.5
    python -m laptop.vision.step_viewer --clip data/clips/<folder>   # detect + watch
    python -m laptop.vision.handcheck   --clip data/clips/<folder>   # mark true strikes
    python tests/hw_stage3_steps.py data/clips/*_slow_walk data/clips/*_normal_walk data/clips/*_jog

Pass criteria (config.yaml stage3_pass), per clip:
  - total count within +-1 of the truth
  - median |timing error| <= 2 frames over matched strikes
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from laptop.config import load_config
from laptop.vision.clip import Clip
from laptop.vision.compare import compare, format_report
from laptop.vision.pose import load_landmarks
from laptop.vision.step_viewer import detect_clip
from laptop.vision.steps import LEGS, SIDE


def heel_vis(lm, leg):
    return lm[f"{SIDE[leg]}_heel_vis"].fillna(0).to_numpy()


def explain(clip, lm, r, cfg):
    """Why each miss/extra probably happened, from landmark visibility."""
    vmin = cfg["steps"]["visibility_min"]
    notes = []
    no_person = float(np.mean(lm["left_hip_x"].isna() & lm["right_hip_x"].isna()))
    if no_person > 0.02:
        notes.append(f"no person found in {100 * no_person:.0f}% of frames")
    for leg in LEGS:
        v = heel_vis(lm, leg)
        hidden = float(np.mean(v < vmin))
        if hidden > 0.05:
            notes.append(f"{leg} heel below visibility {vmin} in {100 * hidden:.0f}% of frames"
                         " (far leg occluded?)")
        for f in r[leg]["missed"]:
            w = v[max(0, f - 5):f + 6]
            if not len(w):
                why = "no landmark data for this frame"
            elif w.min() < vmin:
                why = f"heel hidden (min vis {w.min():.2f})"
            else:
                why = "heel visible -- signal too small/blurred? check thresholds"
            notes.append(f"{leg} MISSED strike at frame {f}: {why}")
        for f in r[leg]["extra"]:
            w = v[max(0, f - 5):f + 6]
            notes.append(f"{leg} EXTRA detection at frame {f} (heel vis {w.mean() if len(w) else 0:.2f})"
                         " -- shuffle/turnaround/jitter?")
    return notes


def main():
    cfg = load_config()
    p = argparse.ArgumentParser()
    p.add_argument("clips", nargs="+")
    p.add_argument("--show", action="store_true", help="watch detection while it runs")
    args = p.parse_args()
    crit, tol = cfg["stage3_pass"], cfg["handcheck"]["match_tol_frames"]

    rows, all_ok = [], True
    for path in args.clips:
        clip = Clip(path)
        truth_path = clip.path("truth.csv")
        print("=" * 72)
        if not truth_path.exists():
            print(f"{clip.name}: no truth.csv -- run: python -m laptop.vision.handcheck --clip {path}")
            all_ok = False
            continue
        det = detect_clip(clip, cfg, display=args.show)
        truth = pd.read_csv(truth_path)
        r = compare([(e.frame_idx, e.leg) for e in det.events],
                    [(int(t.frame_idx), t.leg) for t in truth.itertuples()], tol)
        print(format_report(clip.name, r))
        for note in explain(clip, load_landmarks(clip.landmarks_path(cfg["pose"]["model"])), r, cfg):
            print("  - " + note)
        tot = r["total"]
        count_ok = abs(tot["count_error"]) <= crit["count_tol"]
        timing_ok = tot["median_abs_err"] is not None and tot["median_abs_err"] <= crit["median_timing_frames"]
        ok = count_ok and timing_ok
        all_ok &= ok
        rows.append((clip.name, tot["truth"], tot["detected"], tot["count_error"],
                     tot["missed"], tot["extra"], tot["median_abs_err"], ok))

    print("=" * 72)
    print(f"{'clip':<34} {'true':>4} {'det':>4} {'err':>4} {'miss':>4} {'extra':>5} {'|t| med':>7}  result")
    for name, n_t, n_d, ce, mi, ex, med, ok in rows:
        print(f"{name:<34} {n_t:>4} {n_d:>4} {ce:>+4} {mi:>4} {ex:>5} "
              f"{'--' if med is None else f'{med:.1f}':>7}  {'PASS' if ok else 'FAIL'}")
    print(f"\n(pass: count within +-{crit['count_tol']}, median |timing| <= "
          f"{crit['median_timing_frames']} frames)")
    print(f"STAGE 3 {'PASS' if all_ok and rows else 'FAIL'}")
    sys.exit(0 if all_ok and rows else 1)


if __name__ == "__main__":
    main()
