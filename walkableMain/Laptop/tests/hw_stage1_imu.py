"""
Stage 1 hardware smoke test -- MQTT acquisition from the shin UNO Q.

Opens the live window for --minutes, then prints a PASS/FAIL report and
exits 0 (all pass) or 1. The session is logged to data/sessions/.

Usage (from ME193WalkWearable/, mosquitto running, UNO Q streaming):
    python tests/hw_stage1_imu.py                      # 10-minute clean run
    python tests/hw_stage1_imu.py --minutes 3 --disconnect
                                    # you pull the UNO Q's Wi-Fi/power mid-run;
                                    # checks that each outage is reported as a gap
    python tests/hw_stage1_imu.py --headless --minutes 1   # no window (stats in terminal)

Pass criteria (config.yaml stage1_pass):
  clean run   - >= 99.9% of samples received
              - whole-run rate within 1% of nominal (seq span / laptop time span)
              - clock synced, no malformed messages, one device session
              - plots smooth: median UI refresh >= 80% of ui.refresh_hz
  --disconnect- at least one gap reported, every missing sample accounted for
                by a reported gap; you compare gap durations to what you did
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laptop.acquire.acquirer import ImuAcquirer, format_stats
from laptop.config import load_config


def check(results, ok, label, detail):
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label:<28} {detail}")


def report(s, cfg, disconnect_mode):
    crit = cfg["stage1_pass"]
    print("\n" + "=" * 72)
    print(format_stats(s))
    print("=" * 72)
    results = []
    nominal = s["nominal_rate_hz"]

    if disconnect_mode:
        gaps = s["gaps"]
        check(results, len(gaps) >= 1, "gap(s) reported", f"{len(gaps)} gap(s)")
        accounted = sum(g.still_missing for g in gaps)
        check(results, accounted == s["missing"], "missing == sum of gaps",
              f"missing {s['missing']}, gaps say {accounted}")
        for g in gaps:
            print(f"         gap at seq {g.first_missing}: {g.count} samples "
                  f"= {g.count / nominal:.2f} s  (recovered late: {g.recovered})")
        print("         -> compare these durations to how long you disconnected")
    else:
        frac = s["received_frac"]
        check(results, frac >= crit["min_received_frac"], "samples received",
              f"{100 * frac:.3f}% (need >= {100 * crit['min_received_frac']:.1f}%)"
              f"  missing {s['missing']} of {s['expected']}")
        rate = s["rate_hz_run"]
        err = None if rate is None else abs(rate / nominal - 1)
        check(results, err is not None and err <= crit["rate_tol_frac"], "rate within 1%",
              "no data" if rate is None else
              f"{rate:.3f} Hz vs {nominal:g} Hz ({100 * err:.3f}% off)")
        check(results, len(s["gaps"]) == 0, "no gaps", f"{len(s['gaps'])} gap(s)")

    check(results, s["offset_ns"] is not None, "clock synced",
          f"{s['pongs']} pongs, rtt {s['rtt_ms']} ms, drift {s['drift_ppm']} ppm")
    check(results, s["bad_messages"] == 0, "no malformed messages", f"{s['bad_messages']}")
    check(results, s["sessions_seen"] == 1, "one device session",
          f"{s['sessions_seen']} (device restarted?)" if s["sessions_seen"] != 1 else "1")
    if "ui_fps_median" in s:
        target = cfg["ui"]["refresh_hz"]
        fps = s["ui_fps_median"]
        check(results, fps is not None and fps >= 0.8 * target, "plots smooth",
              f"median {fps and round(fps)} fps, 5th pct {s['ui_fps_p5'] and round(s['ui_fps_p5'])}"
              f" fps (target {target})")
    print(f"  latency median {s['latency_ms_median']} ms, p95 {s['latency_ms_p95']} ms (info only)")
    print(f"  log: {s['log_folder'] or 'see [acquire] saved ... line above'}")
    passed = all(results)
    print(f"\nSTAGE 1 {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    cfg = load_config()
    p = argparse.ArgumentParser()
    p.add_argument("--minutes", type=float, default=cfg["stage1_pass"]["default_duration_s"] / 60)
    p.add_argument("--disconnect", action="store_true", help="deliberate-outage test")
    p.add_argument("--headless", action="store_true")
    args = p.parse_args()
    duration = args.minutes * 60
    result = {}

    if args.headless:
        acq = ImuAcquirer(cfg)
        acq.start()
        t_end = time.monotonic() + duration
        try:
            while time.monotonic() < t_end:
                time.sleep(min(10, max(0.1, t_end - time.monotonic())))
                print(format_stats(acq.stats()), end="\n\n")
        except KeyboardInterrupt:
            print("stopped early")
        result["stats"] = acq.stats()
        acq.stop()
    else:
        from laptop.ui.imu_live import run
        run(duration, on_finish=lambda s: result.setdefault("stats", s), cfg=cfg)

    sys.exit(0 if report(result["stats"], cfg, args.disconnect) else 1)


if __name__ == "__main__":
    main()
