"""
Stage 2 hardware smoke test -- webcam acquisition.

Records --minutes with a live preview (q stops early), then checks the
saved clip and prints PASS/FAIL. Exits 0 (all pass) or 1.

Usage (from ME193WalkWearable/):
    python tests/hw_stage2_webcam.py                 # 5-minute run
    python tests/hw_stage2_webcam.py --minutes 1
    python tests/hw_stage2_webcam.py --check data/clips/<folder>   # re-check a saved clip

Pass criteria (config.yaml stage2_pass):
  - every 10 s window's fps within 5% of the run's mean fps
  - dropped frames <= max_drops (0); each hole is listed with its time so you
    can say whether it was explained (e.g. you covered the lens)
  - frames.csv rows == frames decoded from video.mp4 == frames captured
  - writer kept up (no camera read failures)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import pandas as pd

from laptop.config import load_config
from laptop.vision.frametiming import find_drops, find_late_stamps, fps_stability


def check(results, ok, label, detail):
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label:<26} {detail}")


def count_video_frames(path):
    cap = cv2.VideoCapture(str(path))
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return n


def report(folder, cfg, captured=None, read_failures=0, max_queue=None, stop_reason=None):
    crit, cam = cfg["stage2_pass"], cfg["camera"]
    frames = pd.read_csv(os.path.join(folder, "frames.csv"))
    t = frames["t_ns"].to_numpy()
    print("\n" + "=" * 72 + f"\nclip {folder}")
    print(f"  recorded {(t[-1] - t[0]) / 1e9:.1f} s, {len(t)} frames"
          + (f"  (stopped: {stop_reason})" if stop_reason else ""))
    results = []

    stab = fps_stability(t, crit["fps_window_s"], crit["fps_tol_frac"])
    wf = stab["window_fps"]
    print(f"  mean fps {stab['mean_fps']:.3f} (target {cam['fps']}), "
          f"{len(wf)} windows of {crit['fps_window_s']} s: "
          f"min {wf.min() if len(wf) else float('nan'):.2f}  max {wf.max() if len(wf) else float('nan'):.2f}")
    worst = stab["worst_dev_frac"]
    check(results, stab["stable"], "fps stable within 5%",
          "too short to judge" if worst is None else f"worst window {100 * worst:.2f}% off the mean")
    if stab["mean_fps"] < 0.95 * cam["fps"]:
        print(f"         note: camera delivered {stab['mean_fps']:.1f} fps, below the {cam['fps']} target"
              " -- usually low light (auto-exposure lengthens frames); try more light or --probe")

    drops = find_drops(t, stab["mean_fps"] or cam["fps"], cam["drop_factor"])
    n_drop = sum(d.missing for d in drops)
    check(results, n_drop <= crit["max_drops"], "no dropped frames",
          f"{n_drop} frame(s) lost in {len(drops)} hole(s)")
    for d in drops[:20]:
        print(f"         at {(d.t_ns - t[0]) / 1e9:7.2f} s (after frame {d.after_frame}): "
              f"{d.interval_ms:.0f} ms interval = {d.missing} frame(s)")

    late = find_late_stamps(t, stab["mean_fps"] or cam["fps"], cam["drop_factor"])
    if late:
        worst = max(lag for _, lag in late)
        print(f"  info: {len(late)} frame(s) arrived on time but were stamped late "
              f"(worst {worst:.0f} ms) -- not drops; matters for Stage 4 sync")
    n_video = count_video_frames(os.path.join(folder, "video.mp4"))
    detail = f"csv {len(frames)}, video {n_video}"
    ok = len(frames) == n_video
    if captured is not None:
        detail += f", captured {captured}"
        ok = ok and captured == len(frames)
    check(results, ok, "frame counts match", detail)
    idx_ok = (frames["frame_idx"].diff().dropna() == 1).all() and frames["t_ns"].is_monotonic_increasing
    check(results, bool(idx_ok), "frames.csv in order", "frame_idx consecutive, t_ns increasing")
    check(results, read_failures == 0, "no camera read failures", f"{read_failures}")
    if max_queue is not None:
        print(f"  writer queue peak {max_queue} frames (info: large = disk/encoder struggling)")

    passed = all(results)
    print(f"\nSTAGE 2 {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    cfg = load_config()
    p = argparse.ArgumentParser()
    p.add_argument("--minutes", type=float, default=cfg["stage2_pass"]["default_duration_s"] / 60)
    p.add_argument("--check", default=None, help="only check an existing clip folder")
    args = p.parse_args()

    if args.check:
        sys.exit(0 if report(args.check, cfg) else 1)

    from laptop.vision.capture import CameraRecorder, new_clip_dir, preview_loop
    folder = new_clip_dir(cfg, "stage2_test")
    rec = CameraRecorder(cfg, folder, "stage2_test")
    rec.start()
    print(f"recording {args.minutes:g} min to {folder} (q in the window stops early)")
    reason = preview_loop(rec, args.minutes * 60)
    rec.stop()
    s = rec.stats()
    sys.exit(0 if report(str(folder), cfg, s["frames"], s["read_failures"], s["max_queue"],
                         reason) else 1)


if __name__ == "__main__":
    main()
