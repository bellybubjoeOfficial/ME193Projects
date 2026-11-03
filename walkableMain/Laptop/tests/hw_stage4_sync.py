"""
Stage 4 hardware test -- synced IMU + video recording, stomp alignment.

Records a session (IMU on the shin, camera side-on), then finds the start
and end stomps in both streams and prints PASS/FAIL. Exits 0 on pass.

Usage (from ME193WalkWearable/, mosquitto running, UNO Q streaming):
    python tests/hw_stage4_sync.py --course 10 --leg R            # 5-minute session
    python tests/hw_stage4_sync.py --course 10 --leg R --minutes 1
    python tests/hw_stage4_sync.py --check data/recordings/<folder>   # re-check a session

Protocol: stand still ~2 s -> STOMP the IMU foot (heel visible to the
camera) -> walk the course -> stand still -> STOMP -> press q.

Pass criteria (config.yaml stage4_pass):
  - both stomps found in the IMU and in the video
  - |residual| <= 1 frame at both stomps (after the MQTT clock offset)
  - drift (end - start residual) <= 10 ms
If the residual is larger, sync_correction.json holds the stomp-based
correction that later stages apply.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laptop.config import load_config
from laptop.sync.stomp import analyze, format_report


def check(results, ok, label, detail):
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label:<28} {detail}")


def report(folder, cfg):
    crit = cfg["stage4_pass"]
    out = analyze(folder, cfg)
    meta = json.load(open(os.path.join(folder, "meta.json"))) if os.path.exists(
        os.path.join(folder, "meta.json")) else {}
    print("\n" + "=" * 72 + f"\nsession {folder}")
    if meta:
        print(f"  {meta.get('speed')} | course {meta.get('course_length_m')} m | IMU on {meta.get('imu_leg')}"
              f" | {meta.get('duration_s')} s | IMU {meta['imu']['received']}/{meta['imu']['expected']}"
              f" samples, {meta['imu']['gaps']} gaps | video {meta['video']['frames']} frames"
              f" @ {meta['video']['fps']:.2f} fps, {meta['video']['dropped_frames']} dropped")
    print("  " + format_report(out, cfg).replace("\n", "\n  "))
    results = []
    for name in ("start", "end"):
        s = out["stomps"][name]
        check(results, bool(s["imu"]), f"{name} stomp in IMU",
              f"{s['imu']['peak_g']} g" if s["imu"] else "not found")
        check(results, bool(s["video"]), f"{name} stomp in video",
              f"frame {s['video']['frame_idx']}" if s["video"] else "not found")
        if "residual_frames" in s:
            check(results, abs(s["residual_frames"]) <= crit["max_residual_frames"],
                  f"{name} residual <= {crit['max_residual_frames']} frame",
                  f"{s['residual_ms']:+.1f} ms = {s['residual_frames']:+.2f} frames")
    if "drift_ms" in out:
        check(results, abs(out["drift_ms"]) <= crit["max_drift_ms"], "drift negligible",
              f"{out['drift_ms']:+.1f} ms over {out['span_s']:.0f} s ({out['drift_ppm']:+.0f} ppm)")
    else:
        check(results, False, "drift negligible", "needs both stomps")
    if "correction" in out and not all(results):
        r = out["correction"]["residual_ns"]
        print(f"  -> correction applied from stomps: residual {r[0] / 1e6:+.1f} ms"
              + (f" .. {r[-1] / 1e6:+.1f} ms" if len(r) > 1 else "")
              + " (sync_correction.json); later stages use it")
    print(f"  plot: {os.path.join(folder, 'stomp.png')}")
    passed = all(results)
    print(f"\nSTAGE 4 {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    cfg = load_config()
    p = argparse.ArgumentParser()
    p.add_argument("--check", help="re-check an existing session folder")
    p.add_argument("--course", type=float, help="course length (m)")
    p.add_argument("--leg", choices=["L", "R"])
    p.add_argument("--speed", default="normal")
    p.add_argument("--notes", default="stage 4 test")
    p.add_argument("--minutes", type=float, default=cfg["stage4_pass"]["default_duration_s"] / 60)
    args = p.parse_args()
    if args.check:
        sys.exit(0 if report(args.check, cfg) else 1)
    if args.course is None or args.leg is None:
        p.error("--course and --leg are required to record")
    from laptop.sync.record import record
    folder = record(cfg, args.speed, args.course, args.leg, args.notes, args.minutes,
                    label=f"stage4_{args.speed}")
    sys.exit(0 if report(str(folder), cfg) else 1)


if __name__ == "__main__":
    main()
