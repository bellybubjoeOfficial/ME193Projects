"""
H1.2 -- scripted drive test on the Double Motor (no Xbox needed).

Usage (from ME193Sumo/, robot propped up or on the floor with room):
    python tests/hw_drive.py

Runs: forward → reverse → spin left → spin right, 1 s each, at 40% speed.
Pass when each move matches what's printed and it stops at the end.
If a side goes the wrong way, flip INVERT_LEFT / INVERT_RIGHT in config.py
and rerun.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from drive import apply_inversion, arcade_mix
from lelib import doubleMotor

MOVES = [
    ("FORWARD", 1, 0),
    ("REVERSE", -1, 0),
    ("SPIN LEFT", 0, -1),
    ("SPIN RIGHT", 0, 1),
]


def main():
    print("Connecting to Double Motor...")
    dm = doubleMotor()
    dm.connect(card_serial=config.DRIVE_CARD_SERIAL, card_color=config.DRIVE_CARD_COLOR)
    print(f"Connected. INVERT_LEFT={config.INVERT_LEFT} INVERT_RIGHT={config.INVERT_RIGHT}\n")
    try:
        for label, throttle, turn in MOVES:
            left, right = apply_inversion(*arcade_mix(throttle, turn, max_speed=40, turn_gain=1.0))
            print(f"{label:10}  left={left:+4d}  right={right:+4d}")
            dm.movement_move_tank(left, right)
            time.sleep(1.0)
            dm.stop()
            time.sleep(0.5)
    finally:
        dm.stop()
        print("\nDone -- did every move match its label?")


if __name__ == "__main__":
    main()
