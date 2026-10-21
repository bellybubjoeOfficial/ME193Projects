"""
H2.2 / H5.1 -- controller sounds + rumble (no LEGO hardware needed).

Usage (from ME193Sumo/):
    python tests/hw_alert.py

  X → the edge alert main.py fires when the color sensor hits the tape:
      pass when you hear the double beep AND feel the controller buzz
  Y → sad tune ("wah wah wah wahhh")
  A → victory fanfare
Holding a button plays it once; pressing again restarts the tune.
If rumble prints "not supported", the pad/OS combo can't rumble over
Bluetooth -- try a USB cable or update the controller firmware.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from sounds import SoundEffects
from xbox import ButtonEdges, XboxController


def main():
    pad = XboxController()
    print("Waiting for controller...")
    pad.connect()
    sfx = SoundEffects()
    print(f"Connected: {pad.name}. X = edge alert, Y = sad, A = victory (Ctrl+C to quit).")
    buttons = ButtonEdges()
    try:
        while True:
            pressed = buttons.update(pad.read())
            if "x" in pressed:
                sfx.play_edge()
                ok = pad.rumble(config.RUMBLE_STRENGTH, config.RUMBLE_MS)
                print(f"edge alert! rumble {'ok' if ok else 'not supported'}")
            if "y" in pressed:
                sfx.play_sad()
                print("sad")
            if "a" in pressed:
                sfx.play_victory()
                print("victory")
            time.sleep(0.02)
    except KeyboardInterrupt:
        print()
    finally:
        pad.close()


if __name__ == "__main__":
    main()
