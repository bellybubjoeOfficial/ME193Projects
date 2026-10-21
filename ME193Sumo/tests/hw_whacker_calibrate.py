"""
H4.5 -- find the whacker angles where it blocks the distance sensor.

Usage (from ME193Sumo/):
    python tests/hw_whacker_calibrate.py                 # sweep ±90° around base, save
    python tests/hw_whacker_calibrate.py --from 0 --to 180
    python tests/hw_whacker_calibrate.py --verify        # live check of the saved file

Before running:
  - the UNO Q publisher is running (hw_unoq.py passes)
  - the whacker sits at its CENTER position (the sweep is ±90° around it;
    main.py then rests the arm at center + 90° and strikes to center - 90°)
  - nothing within ~50 cm in front of the robot (so anything the sensor
    sees during the sweep must be the whacker itself)
  - the whole sweep range is physically free (the arm won't hit the chassis)

It steps the whacker slowly from base+FROM to base+TO and back, holding
each step for a moment to collect distance readings. Every angle where the
sensor sees something within WHACKER_BLIND_DETECT_MM becomes a blind zone
(padded by WHACKER_BLIND_MARGIN_DEG) in whacker_calibration.json. Then it
returns to base.

Pass when: the reported blind zone(s) match where the arm visibly crosses
in front of the sensor, and --verify shows BLIND exactly there.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import legoeducation as le

import config
from lelib import singleMotor
from unoq_link import UnoQLink
from whacker import (CALIBRATION_FILE, SweepSample, blind_arcs_from_sweep, in_blind_zone,
                     load_blind_calibration, save_blind_calibration, sees_something)

ARRIVE_TOL_DEG = 2
STEP_TIMEOUT_S = 1.5


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--from", dest="start", type=int, default=-90, help="degrees from base (default -90)")
    p.add_argument("--to", dest="end", type=int, default=90, help="degrees from base (default +90)")
    p.add_argument("--step", type=int, default=3, help="degrees per step (default 3)")
    p.add_argument("--speed", type=int, default=20, help="sweep speed 0..100 (default 20)")
    p.add_argument("--dwell", type=float, default=0.15, help="seconds of readings per step")
    p.add_argument("--verify", action="store_true", help="only show a live check of the saved file")
    return p.parse_args()


def move_to(sm, target, speed):
    """Hub-side position move; returns False if the arm couldn't get there."""
    sm.motor_run_to_relative_position(target, speed=speed, blocking=False)
    deadline = time.monotonic() + STEP_TIMEOUT_S
    while time.monotonic() < deadline:
        if abs(sm.motor.position - target) <= ARRIVE_TOL_DEG:
            return True
        time.sleep(0.01)
    return False


def closest_reading(link, seconds):
    """Smallest valid distance seen over `seconds`, or None."""
    best = None
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        mm = link.distance_mm()
        if mm is not None and (best is None or mm < best):
            best = mm
        time.sleep(0.02)
    return best


def sweep(sm, link, args):
    home = sm.motor.position
    forward = list(range(home + args.start, home + args.end + 1, args.step))
    path = forward + forward[::-1]
    samples = []
    print(f"\nSweeping base{args.start:+d}° → base{args.end:+d}° and back "
          f"({len(path)} steps, speed {args.speed})\n")
    for i, target in enumerate(path):
        if not move_to(sm, target, args.speed):
            print(f"\n!! couldn't reach {target - home:+d}° from base (arm blocked?) -- recording anyway")
        mm = closest_reading(link, args.dwell)
        sample = SweepSample(abs_deg=sm.motor.absolutePosition, mm=mm)
        samples.append(sample)
        flag = "SEES WHACKER" if sees_something(mm) else ""
        print(f"\r  step {i + 1:3}/{len(path)}  base{target - home:+4d}°  abs={sample.abs_deg:4}°  "
              f"distance={('---' if mm is None else mm):>5}  {flag:12}", end="", flush=True)
    print()
    move_to(sm, home, args.speed)
    return samples, home


def verify(sm, link):
    arcs = load_blind_calibration()
    if arcs is None:
        print(f"No {CALIBRATION_FILE} yet -- run without --verify first.")
        return
    sm.motor_set_end_state(le.MOTOR_END_STATE_COAST)
    sm.motor_stop()
    print("Blind zones:", ", ".join(f"{a.start:.0f}°→{a.end:.0f}°" for a in arcs) or "none")
    print("Move the whacker by hand (motor is slack). Ctrl+C to finish.\n")
    while True:
        a = sm.motor.absolutePosition
        mm = link.distance_mm()
        state = "BLIND (distance ignored)" if in_blind_zone(a, arcs) else "clear   (distance used)"
        print(f"\r  abs={a:4}°  {state}  raw distance={('---' if mm is None else mm):>5}   ",
              end="", flush=True)
        time.sleep(0.05)


def main():
    args = parse_args()
    print(f"Connecting to UNO Q broker {config.MQTT_BROKER}...")
    link = UnoQLink()
    link.connect()
    time.sleep(1.0)
    if link.stats("distance")["count"] == 0:
        print("!! No distance messages from the UNO Q -- is the publisher running? (hw_unoq.py)")
        return 1
    print("Connecting to Single Motor (whacker)...")
    sm = singleMotor()
    sm.connect(card_serial=config.WHACKER_CARD_SERIAL, card_color=config.WHACKER_CARD_COLOR)
    try:
        if args.verify:
            verify(sm, link)
            return 0
        sm.motor_set_end_state(le.MOTOR_END_STATE_HOLD)
        input("Put the whacker at its CENTER position, clear ~50 cm in front of the robot, "
              "then press Enter...")
        center_abs = sm.motor.absolutePosition   # where the arm sits now = center
        samples, home = sweep(sm, link, args)
        arcs = blind_arcs_from_sweep(samples)
        save_blind_calibration(arcs, samples, center_abs=center_abs,
                               sweep_from=args.start, sweep_to=args.end,
                               step=args.step, detect_mm=config.WHACKER_BLIND_DETECT_MM,
                               margin_deg=config.WHACKER_BLIND_MARGIN_DEG)
        hits = sum(sees_something(x.mm) for x in samples)
        print(f"\n{hits}/{len(samples)} steps saw the whacker.")
        if arcs:
            for a in arcs:
                print(f"  blind zone: abs {a.start:.0f}° → {a.end:.0f}°  ({a.width:.0f}° incl. margin)")
        else:
            print("  no blind zone found -- the whacker never crossed the sensor in this range")
        print(f"Saved {CALIBRATION_FILE}")
        print("Check it with:  python tests/hw_whacker_calibrate.py --verify")
    except KeyboardInterrupt:
        print()
    finally:
        sm.stop()
        link.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
