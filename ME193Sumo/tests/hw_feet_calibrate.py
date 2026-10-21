"""
H3.1 -- find each foot's base-position encoder range and save it.

Usage (from ME193Sumo/):
    python tests/hw_feet_calibrate.py            # calibrate, then live check
    python tests/hw_feet_calibrate.py --verify   # live check of the saved file only

For each foot (left, then right):
  1. turn the foot to ONE end of its flat/base position → press Enter
  2. turn it THROUGH the base position to the other end → press Enter
     (the direction you turn tells the script which side of the circle
     is the base range)
The live encoder angle is printed while you turn. The motors are left
slack so you can turn the feet by hand; if a foot is too stiff, type
"+" or "-" (optionally with degrees, e.g. "+ 20") then Enter to jog it.

Saves feet_calibration.json, then shows a live IN BASE / out readout so
you can check the saved ranges (Ctrl+C to finish).

Pass when: the readout says IN BASE (or IN TARGET, its center 50%) exactly
when each foot looks flat,
and each range is wider than ~15° (narrower and a moving foot can skip
straight over it between two control-loop ticks -- lower SETTLE_SPEED or
re-mark a wider range).
"""

import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import legoeducation as le

import config
from feet import CALIBRATION_FILE, arc_from_marks, load_calibration, save_calibration
from lelib import doubleMotor

SIDES = (("LEFT", le.MOTOR_LEFT), ("RIGHT", le.MOTOR_RIGHT))
DEFAULT_JOG_DEG = 10


def angle(dm, motor):
    return dm.motor[motor].absolutePosition


class LivePrinter:
    """Prints one foot's live angle on a single line while input() waits."""

    def __init__(self, dm):
        self.dm = dm
        self.motor = None
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while not self._stop.is_set():
            if self.motor is not None:
                print(f"\r    angle = {angle(self.dm, self.motor):4}°   "
                      f"(Enter = save, +/- [deg] = jog) > ", end="", flush=True)
            time.sleep(0.2)

    def stop(self):
        self._stop.set()


def mark(dm, motor, printer, prompt):
    print(prompt)
    printer.motor = motor
    while True:
        cmd = input().strip()
        if cmd == "":
            printer.motor = None
            value = angle(dm, motor)
            print(f"\n    saved {value}°")
            return value, dm.motor[motor].position
        if cmd[0] in "+-":
            deg = int(cmd[1:].strip() or DEFAULT_JOG_DEG)
            direction = (le.MOTOR_MOVE_DIRECTION_CLOCKWISE if cmd[0] == "+"
                         else le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE)
            dm.motor_run_for_degrees(deg, direction=direction, motor=motor, speed=20)
            dm.motor_set_end_state(le.MOTOR_END_STATE_COAST, motor=motor)
            dm.motor_stop(motor=motor)


def calibrate(dm):
    printer = LivePrinter(dm)
    arcs = []
    for label, motor in SIDES:
        print(f"\n=== {label} foot ===")
        a, a_count = mark(dm, motor, printer, f"Turn the {label} foot to ONE end of its base position.")
        b, b_count = mark(dm, motor, printer, f"Turn the {label} foot to the OTHER end of its base position,\n"
                                              f"    turning it THROUGH the base position (not back around the other way).")
        arc = arc_from_marks(a, b, travelled=b_count - a_count)
        print(f"    {label} base range: {arc.start:.0f}° → {arc.end:.0f}°  (width {arc.width:.0f}°)")
        if arc.width < 15:
            print("    !! narrow range -- a moving foot may skip over it; consider re-marking wider")
        arcs.append(arc)
    printer.stop()
    save_calibration(*arcs)
    print(f"\nSaved {CALIBRATION_FILE}")


def status(arc, a):
    if arc.middle(config.SETTLE_TARGET_FRACTION).contains(a):
        return "IN TARGET"
    return "IN BASE  " if arc.contains(a) else "out      "


def verify(dm):
    left_arc, right_arc = load_calibration()
    print("\nLive check -- turn the feet and watch (Ctrl+C to finish)")
    print("IN TARGET = center of the base range, where settling stops.\n")
    while True:
        l, r = angle(dm, le.MOTOR_LEFT), angle(dm, le.MOTOR_RIGHT)
        line = f"LEFT {l:4}° {status(left_arc, l)}   RIGHT {r:4}° {status(right_arc, r)}"
        print("\r" + line, end="", flush=True)
        time.sleep(0.1)


def main():
    print("Connecting to Double Motor...")
    dm = doubleMotor()
    dm.connect(card_serial=config.DRIVE_CARD_SERIAL, card_color=config.DRIVE_CARD_COLOR)
    for _, motor in SIDES:   # let the feet turn freely by hand
        dm.motor_set_end_state(le.MOTOR_END_STATE_COAST, motor=motor)
        dm.motor_stop(motor=motor)
    try:
        if "--verify" not in sys.argv:
            calibrate(dm)
        verify(dm)
    except KeyboardInterrupt:
        print()
    finally:
        dm.stop()


if __name__ == "__main__":
    main()
