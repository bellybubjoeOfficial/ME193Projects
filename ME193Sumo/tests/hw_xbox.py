"""
H1.1 -- live Xbox controller readout (no LEGO hardware needed).

Usage (from ME193Sumo/):
    python tests/hw_xbox.py

Pass when:
  - sticks centered read 0.00 (deadzone works)
  - stick up = +ly, stick right = +lx, triggers go 0 → 1
  - every button lights up when pressed
  - turning the controller off shows connected=False
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xbox import BUTTONS, XboxController


def main():
    pad = XboxController()
    print("Waiting for controller...")
    pad.connect()
    print(f"Connected: {pad.name}   (Ctrl+C to quit)\n")
    try:
        while True:
            s = pad.read()
            pressed = " ".join(n.upper() for n in BUTTONS if getattr(s, n)) or "-"
            line = (f"connected={s.connected!s:5}  "
                    f"L({s.lx:+.2f},{s.ly:+.2f})  R({s.rx:+.2f},{s.ry:+.2f})  "
                    f"LT {s.lt:.2f}  RT {s.rt:.2f}  buttons: {pressed}")
            print("\r" + line.ljust(110), end="", flush=True)
            if not s.connected:
                pad.connect(timeout_s=0.1)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print()
    finally:
        pad.close()


if __name__ == "__main__":
    main()
