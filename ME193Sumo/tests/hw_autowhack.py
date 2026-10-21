"""
H4.3 -- whacker base / strike / return bench test (UNO Q + whacker, no feet).

Runs the real whacker sub-tree (policy.decide_whacker), wired like main.py.

Usage (from ME193Sumo/):
    python tests/hw_autowhack.py          # no controller: strikes + auto return only
    python tests/hw_autowhack.py --xbox   # also test RT/LT override + the 2 s hold

Pass when:
  - on start the arm moves to BASE (center + 90°, on the right) and holds
  - hand within WHACK_TRIGGER_MM → ONE strike: sweeps through center to the
    far end, then comes straight back to base
  - holding your hand there doesn't strike again; pulling back past
    WHACK_REARM_MM and coming in again gives one more
  - the arm passing in front of the sensor (BLIND) never triggers a strike
  - stopping the Q never causes a strike
  - with --xbox: RT/LT take over at any time; holding RT stops exactly at
    the strike end, holding LT exactly at base (never past either); after
    letting go, the arm stays put for ~2 s, then returns to base
If base ends up on the left, flip WHACKER_BASE_SIDE in config.py.
"""

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import legoeducation as le

import config
from lelib import singleMotor
from policy import Inputs, decide_whacker
from unoq_link import UnoQLink
from whacker import (HOME, STRIKE, ArmGeometry, ArmMotion, ProximityTrigger, center_relative,
                     in_blind_zone, load_blind_calibration, load_center_abs, manual_move,
                     visible_distance)
from xbox import XboxController, XboxState


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xbox", action="store_true", help="use the Xbox triggers for manual override")
    args = p.parse_args()

    center_abs = load_center_abs()
    blind_arcs = load_blind_calibration()
    if center_abs is None or blind_arcs is None:
        print("!! no whacker_calibration.json -- run tests/hw_whacker_calibrate.py first")
        return 1

    pad = None
    if args.xbox:
        pad = XboxController()
        print("Waiting for controller...")
        pad.connect()
    print(f"Connecting to UNO Q broker {config.MQTT_BROKER}...")
    link = UnoQLink()
    link.connect()
    print("Connecting to Single Motor (whacker)...")
    sm = singleMotor()
    sm.connect(card_serial=config.WHACKER_CARD_SERIAL, card_color=config.WHACKER_CARD_COLOR)
    sm.motor_set_end_state(le.MOTOR_END_STATE_HOLD)
    while math.isnan(sm.motor.position) or math.isnan(sm.motor.absolutePosition):
        time.sleep(0.05)

    geometry = ArmGeometry(center=center_relative(sm.motor.position, sm.motor.absolutePosition, center_abs))
    motion = ArmMotion(geometry)
    prox = ProximityTrigger()
    print(f"center abs {center_abs:.0f}°  →  base {geometry.base}, strike end {geometry.strike_end} "
          f"(relative counter, now {sm.motor.position})")
    print(f"Strike within {config.WHACK_TRIGGER_MM} mm, re-arm past {config.WHACK_REARM_MM} mm "
          f"(Ctrl+C to quit)\n")

    last_manual = -math.inf
    last_mode = None
    last_target = None
    last_speed = None
    strikes = 0
    try:
        while True:
            now = time.monotonic()
            pos = sm.motor.position
            abs_ = sm.motor.absolutePosition
            raw_mm = link.distance_mm()
            blind = in_blind_zone(abs_, blind_arcs)
            entered = prox.update(visible_distance(raw_mm, abs_, blind_arcs))
            target = motion.step(pos, now) if motion.active else None

            w = decide_whacker(Inputs(
                pad=pad.read() if pad else XboxState(connected=True),
                opponent_entered=entered,
                arm_phase=motion.phase,
                arm_target=target,
                arm_at_base=geometry.at_base(pos),
                manual_idle_s=now - last_manual,
            ))

            if w["arm_command"] == "cancel":
                motion.cancel()
            elif w["arm_command"] in (STRIKE, HOME):
                motion.start(w["arm_command"], now)
                if w["arm_command"] == STRIKE:
                    strikes += 1
                print(f"\n!! {w['arm_command']}" + (f" #{strikes} at {raw_mm} mm" if w["arm_command"] == STRIKE else ""))
            if w["whack_mode"] == "manual":
                last_manual = now

            move = None
            if w["whack_mode"] in (STRIKE, HOME):
                move = (w["whack_target"], config.WHACKER_SPEED)
            elif w["whack_mode"] == "manual":
                move = manual_move(w["whack"], geometry)       # can't pass the end points
            if move:
                if move != last_target:
                    sm.motor_run_to_relative_position(move[0], speed=move[1], blocking=False)
                    last_target = move
                last_speed = None
            else:
                last_target = None
                if w["whack"] != last_speed:
                    if w["whack"] == 0:
                        sm.stop()
                    else:
                        sm.run(w["whack"])
                    last_speed = w["whack"]

            if w["whack_mode"] != last_mode and w["whack_mode"] in ("off", "manual"):
                print(f"\n   [{w['whack_mode']}] at {pos} ({'BASE' if geometry.at_base(pos) else 'not base'})")
            last_mode = w["whack_mode"]

            print(f"\rdistance={('---' if raw_mm is None else raw_mm):>5} mm "
                  f"{'BLIND' if blind else '     '}  {w['whack_mode']:12}  "
                  f"{'armed' if prox.armed else 'wait re-arm'}  pos={pos:6} abs={abs_:4}  "
                  f"strikes={strikes}   ", end="", flush=True)
            time.sleep(max(0.0, 1 / config.LOOP_HZ - (time.monotonic() - now)))
    except KeyboardInterrupt:
        print()
    finally:
        sm.stop()
        link.close()
        if pad:
            pad.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
