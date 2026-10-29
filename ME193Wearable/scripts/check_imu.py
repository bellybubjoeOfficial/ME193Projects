#!/usr/bin/env python3
"""Stage 1: run the IMU connection checks on the latest check capture.

    python scripts/check_imu.py                    # uses data/check/
    python scripts/check_imu.py --dir path/to/check

Needs the app to have run long enough to finish its capture (CHECK_SECONDS,
default 30 s) with the board sitting still. Exit code 0 = no FAILs.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wearable import checks, config, imu

TROUBLESHOOTING = """
Troubleshooting checklist
-------------------------
 1. Is the right sketch running?  arduino-app-cli app list  (activity_wearable = running)
    and  arduino-app-cli monitor  should print a line every second.
 2. Wiring (power the board OFF while rewiring):
      VCC -> 5V only if your breakout has its own 3.3 V regulator (GY-521 does),
             otherwise VCC -> 3.3V.  GND -> GND.
      SDA -> A4, SCL -> A5 (bus Wire2). The SDA/SCL header pins next to
      AREF are damaged on this board and are no longer scanned.
      SDA and SCL swapped is the #1 mistake: the scan then finds nothing.
 3. Pull-ups: I2C needs pull-up resistors on SDA and SCL. GY-521 boards have
    them on-board. A bare chip needs ~4.7 kΩ from SDA and from SCL to 3.3 V.
 4. Address: 0x68 if AD0 is low/floating, 0x69 if AD0 is tied high. The sketch
    tries both. If the scan finds a device at another address, it's not the IMU.
 5. Voltage: with the board powered, measure SDA->GND and SCL->GND with a
    multimeter. Both should idle at ~3.3 V. If you read ~5 V, the pull-ups go
    to 5 V: stop and ask before continuing (STM32 analog pins A4/A5 are not
    5 V tolerant).
 6. Loose jumpers: breadboard + dupont wires are the usual cause of
    intermittent i2c_err/s in the monitor, or of dropped samples.
 7. |a| far from 1 g or gyro bias > 20 °/s with the board still: the board
    moved, or the chip is damaged/counterfeit. Check WHO_AM_I.
 8. Sample rate low / dropped samples while wiring is fine: check
    `arduino-app-cli app logs ... --follow` for errors, and whether another
    heavy process is running on the Q.
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=config.CHECK_DIR)
    args = ap.parse_args()

    info_path = args.dir / "imu_info.json"
    csv_path = args.dir / "imu_check.csv"
    if not info_path.exists():
        print(f"Missing {info_path}: has the app run with the Stage 1 sketch?")
        return 2
    info = json.loads(info_path.read_text())
    df = imu.load_csv(csv_path) if csv_path.exists() else None
    if df is None and (args.dir / "imu_check.csv.partial").exists():
        print(f"Capture still in progress ({config.CHECK_SECONDS} s); wait and re-run.")

    print(f"Sketch reported: {info['raw']}\n")
    results = checks.run_all(info, df)
    width = max(len(r.name) for r in results)
    for r in results:
        print(f"  [{r.status}] {r.name:<{width}}  {r.detail}")

    failed = [r for r in results if not r.ok]
    print()
    if failed:
        print(f"{len(failed)} check(s) FAILED.")
        print(TROUBLESHOOTING)
        return 1
    print("All Stage 1 checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
