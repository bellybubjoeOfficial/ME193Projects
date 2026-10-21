"""
H2.1 -- color sensor calibration readout (no motors, no Xbox).

Usage (from ME193Sumo/):
    python tests/hw_color.py

Mount the sensor at its real height on the robot, then slide the robot
over: the ring surface, the blue tape, and the floor outside the ring.

Pass when:
  - TAPE? reads YES on the tape and no on both other surfaces
  - it stays steady (no flicker) while sitting still on each surface
If not, change TAPE_COLOR_NAMES in config.py (e.g. add "Azure" or
"Teal" if that's what the tape reads as) and rerun.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from lelib import colorSensor
from ring import is_tape


def main():
    print("Connecting to Color Sensor...")
    cs = colorSensor()
    cs.connect(card_serial=config.COLOR_CARD_SERIAL, card_color=config.COLOR_CARD_COLOR)
    print(f"Connected. Tape colors = {config.TAPE_COLOR_NAMES}   (Ctrl+C to quit)\n")
    try:
        while True:
            name = cs.detect_color()
            r, g, b = cs.raw_rgb()
            line = (f"TAPE? {'YES' if is_tape(name) else 'no ':3}   color={name:9}  "
                    f"reflection={cs.reflection():3}  rgb=({r:5},{g:5},{b:5})")
            print("\r" + line.ljust(90), end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
