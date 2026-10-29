#!/usr/bin/env python3
"""Stage 3: copy recordings from the UNO Q to this laptop (run ON THE LAPTOP).

    python scripts/fetch_data.py                          # from arduino@UNOQ3.local
    python scripts/fetch_data.py --host 10.247.137.61     # if .local names don't resolve
    python scripts/fetch_data.py --validate               # then validate + plot what arrived

Uses scp (built into macOS, Linux and Windows 10+), so the laptop must be on
the same network as the Q. Copies data/raw/*.csv and data/sessions.csv into
this project's data/ folder. Files already here are copied again (cheap: a
60 s recording is ~400 kB). Unfinished .csv.partial files are skipped.
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wearable import config

REMOTE_APP = "~/ArduinoApps/activity_wearable"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="UNOQ3.local", help="board hostname or IP")
    ap.add_argument("--user", default="arduino")
    ap.add_argument("--validate", action="store_true", help="run validate_recording.py --all --plot afterwards")
    args = ap.parse_args()

    remote = f"{args.user}@{args.host}:{REMOTE_APP}/data"
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    steps = [
        (["scp", f"{remote}/raw/{config.FILE_PREFIX}_*.csv", str(config.RAW_DIR)], "recordings"),
        (["scp", f"{remote}/{config.SESSIONS_CSV.name}", str(config.DATA_DIR)], "sessions.csv"),
    ]
    for cmd, what in steps:
        print(f"Copying {what}: {' '.join(cmd)}")
        if subprocess.run(cmd).returncode != 0:
            print(f"scp failed for {what}. Check: same Wi-Fi as the Q? `ssh {args.user}@{args.host}` works? "
                  "Try --host <IP> (on the Q: hostname -I).")
            return 1
    n = len(list(config.RAW_DIR.glob(f"{config.FILE_PREFIX}_*.csv")))
    print(f"\n{n} recording(s) in {config.RAW_DIR}")
    if args.validate:
        script = Path(__file__).with_name("validate_recording.py")
        return subprocess.run([sys.executable, str(script), "--all", "--plot"]).returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
