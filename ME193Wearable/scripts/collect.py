#!/usr/bin/env python3
"""Stage 4: data collection menu. Run ON THE Q (over SSH) with the App running.

    python scripts/collect.py                       # 120 s recordings, 10 s countdown
    python scripts/collect.py --seconds 90 --countdown 15

Loop:  pick an activity (ENTER = same as last time)
       -> 10 s countdown (LED blue, matrix counts down: get into position)
       -> records for --seconds and stops by itself (LED red, matrix shows
          seconds left + progress bar); press ENTER to stop early
       -> beep, quick validation, optional note (or 'x' to throw it away)
       -> summary table, back to the menu

The App does the recording. If SSH drops mid-recording, the recording still
finishes and is logged (without a note); just reconnect and start this again.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from record import print_validation, wait_for  # noqa: E402  (scripts/record.py)
from wearable import config, recording  # noqa: E402


def menu_line():
    return "  ".join(f"{i}) {a}" for i, a in enumerate(config.RECORD_ACTIVITIES, 1))


def print_summary():
    print("\nCollected so far (usable = after trimming "
          f"{config.TRIM_SECONDS} s from each end):")
    print(recording.format_summary(recording.read_sessions(config.SESSIONS_CSV)))


def ask_note(st):
    """Optional note, or discard. Returns nothing; updates sessions.csv."""
    note = input("Note (who/speed/'strap slipped'; ENTER = none; x = discard this recording): ").strip()
    if note.lower() == "x":
        if input(f"Delete {st['file']}? [y/N] ").strip().lower() == "y":
            recording.discard_recording(config.RAW_DIR, config.SESSIONS_CSV, st["file"])
            print("Deleted.")
        return
    if note:
        recording.update_session(config.SESSIONS_CSV, st["file"], note=note)


def record_once(activity, args):
    if (config.CONTROL_DIR / recording.REQUEST_NAME).exists():
        print("An older request is still queued (App not running?). Removing it.")
        (config.CONTROL_DIR / recording.REQUEST_NAME).unlink()
    req = recording.make_request(activity, args.seconds, args.countdown)
    recording.write_request(config.CONTROL_DIR, req)
    print(f"\n{activity}: {args.countdown:.0f} s countdown, then {args.seconds:.0f} s. "
          "Press ENTER any time to stop.")
    try:
        st = wait_for(req, stop_on_enter=True)
    except KeyboardInterrupt:
        print("\nStopped watching; the App keeps going. Restart collect.py to continue.")
        raise
    if st is None:
        (config.CONTROL_DIR / recording.REQUEST_NAME).unlink(missing_ok=True)
        print("Request withdrawn. Start the App, then try again.")
        return
    if st["state"] == "cancelled":
        print("Cancelled during the countdown; nothing recorded.")
        return
    if st["state"] == "error":
        print(f"App error: {st.get('error')}")
        return

    print_validation(st, activity)
    if st["seconds"] < recording.min_useful_seconds():
        print(f"  ! Only {st['seconds']:.0f} s: after trimming there's not even one "
              f"{config.WINDOW_SAMPLES / config.SAMPLE_RATE_HZ:.2f} s window. Consider discarding (x).")
    ask_note(st)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=config.DEFAULT_RECORD_SECONDS)
    ap.add_argument("--countdown", type=float, default=config.DEFAULT_COUNTDOWN_SECONDS)
    args = ap.parse_args()

    print("=== activity_wearable data collection ===  (Ctrl-C or q to quit)")
    print_summary()
    try:
        while True:
            last = recording.load_last_choice(config.CONTROL_DIR)
            print(f"\n{menu_line()}")
            choice = recording.parse_choice(input(f"Activity [ENTER = {last}, q = quit]: "), last)
            if choice is None:
                print("Not understood: type a number, a name, or just ENTER.")
                continue
            if choice == recording.QUIT:
                break
            recording.save_last_choice(config.CONTROL_DIR, choice)
            record_once(choice, args)
            print_summary()
    except (KeyboardInterrupt, EOFError):
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
