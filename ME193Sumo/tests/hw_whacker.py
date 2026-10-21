"""
H3.3 -- whacker on the triggers (Xbox + Single Motor only).

Usage (from ME193Sumo/):
    python tests/hw_whacker.py

RT swings toward the strike end, LT toward base; harder pull = faster.
Release = stop. Once calibrated, the arm can't go past either end point.

Pass when: RT and LT swing opposite ways, speed follows the trigger, it
stops and holds as soon as both are released, and holding a trigger fully
stops exactly at that end point (never past it).
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math

import legoeducation as le

import config
from lelib import singleMotor
from whacker import ArmGeometry, center_relative, load_center_abs, manual_move, whacker_speed
from xbox import XboxController


def main():
    pad = XboxController()
    print("Waiting for controller...")
    pad.connect()
    print("Connecting to Single Motor...")
    sm = singleMotor()
    sm.connect(card_serial=config.WHACKER_CARD_SERIAL, card_color=config.WHACKER_CARD_COLOR)
    sm.motor_set_end_state(le.MOTOR_END_STATE_HOLD)
    while math.isnan(sm.motor.position) or math.isnan(sm.motor.absolutePosition):
        time.sleep(0.05)

    center_abs = load_center_abs()
    geometry = None
    if center_abs is None:
        print("!! not calibrated -- RT/LT run with NO end-point limits")
    else:
        geometry = ArmGeometry(center=center_relative(sm.motor.position, sm.motor.absolutePosition,
                                                      center_abs))
        print(f"Limits: base {geometry.base} (LT end) ... strike end {geometry.strike_end} (RT end)")
    print("Ready. RT / LT swing the whacker (Ctrl+C to quit).\n")
    last = None
    try:
        while True:
            s = pad.read()
            speed = whacker_speed(s.rt, s.lt) if s.connected else 0
            cmd = manual_move(speed, geometry) if geometry else speed
            if cmd != last:
                if not cmd:
                    sm.stop()
                elif geometry:
                    sm.motor_run_to_relative_position(cmd[0], speed=cmd[1], blocking=False)
                else:
                    sm.run(cmd)
                last = cmd
            m = sm.motor
            print(f"\rRT {s.rt:.2f}  LT {s.lt:.2f}  speed={speed:+4d}  "
                  f"position={m.position:6}  absolute={m.absolutePosition:4}   ",
                  end="", flush=True)
            time.sleep(1 / config.LOOP_HZ)
    except KeyboardInterrupt:
        print()
    finally:
        sm.stop()
        pad.close()


if __name__ == "__main__":
    main()
