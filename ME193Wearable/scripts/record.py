#!/usr/bin/env python3
"""Stage 3: ask the running App to make ONE recording, and show its progress.
(For collecting lots of data, use scripts/collect.py: it has a menu.)

Run ON THE Q (over SSH), with the activity_wearable App already running:

    python scripts/record.py sit                       # 120 s, 10 s countdown
    python scripts/record.py sit --seconds 60 --countdown 20 --note "desk chair, feet flat"
    python scripts/record.py sit --no-wait             # leave the request and exit

The App does the actual recording, so it carries on if SSH drops: Ctrl-C or a
lost connection here only stops the progress display. Watch LED4 / the LED
matrix instead: blue + number = countdown, red + bar = recording,
green + check = done, purple + X = error.
Afterwards the file is validated (same checks as scripts/validate_recording.py).
"""

import argparse
import select
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wearable import config, imu, recording, validate

PICKUP_TIMEOUT_S = 5        # the App polls once per second
FINAL_STATES = ("done", "cancelled", "error")


def _enter_pressed(timeout):
    """Wait up to `timeout` s for a line on stdin. True if ENTER was pressed."""
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    return bool(ready) and sys.stdin.readline() != ""


def wait_for(req, stop_on_enter=False):
    """Print status until the App finishes this request. Returns the final status.

    Returns None if the App never picked the request up (request left queued).
    With stop_on_enter, pressing ENTER asks the App to stop early / cancel.
    """
    t_req = req["requested_at"]
    picked_up = False
    stop_sent = False
    last_line, last_state = "", None
    while True:
        if stop_on_enter and not stop_sent and _enter_pressed(0.5):
            recording.write_stop(config.CONTROL_DIR)
            stop_sent = True
            print("\n  stopping…", flush=True)
        elif not stop_on_enter or stop_sent:
            time.sleep(0.5)
        st = recording.read_status(config.CONTROL_DIR) or {}
        if not picked_up:
            if (config.CONTROL_DIR / recording.REQUEST_NAME).exists():
                if time.time() - t_req > PICKUP_TIMEOUT_S:
                    print("The App hasn't picked up the request. Is it running?\n"
                          "  arduino-app-cli app list          (activity_wearable should be 'running')\n"
                          "  arduino-app-cli app logs ~/ArduinoApps/activity_wearable --tail 20")
                    return None
                continue
            picked_up = True
        if st.get("updated_at", 0) <= t_req:
            continue
        state = st["state"]
        if state != last_state and state in ("recording", "done"):
            print("\a", end="", flush=True)                  # terminal beep
        last_state = state
        if state == "countdown":
            line = f"  countdown: {st['remaining_s']:.0f} s — get into position"
        elif state == "recording":
            line = f"  recording {st['file']}: {st['elapsed_s']:.0f} / {st['seconds']:.0f} s, dropped {st['dropped']}"
        elif state in FINAL_STATES:
            print()
            return st
        else:
            line = f"  {state}"
        if stop_on_enter and not stop_sent:
            line += "   (ENTER = stop)"
        if line != last_line:
            print("\r" + line.ljust(90), end="", flush=True)
            last_line = line


def print_validation(st, activity):
    """Validate the file the App just finished. Returns True if no FAILs."""
    path = config.RAW_DIR / st["file"]
    early = " (stopped early)" if st.get("stopped_early") else ""
    # An App started before Stage 4 doesn't report "seconds"; work it out from the sample count.
    st.setdefault("seconds", st["samples"] / config.SAMPLE_RATE_HZ)
    print(f"Saved {path.name}: {st['seconds']:.1f} s{early}, {st['samples']} samples, {st['dropped']} dropped")
    expected = recording.expected_seconds(config.SESSIONS_CSV, st["file"])
    results = validate.run_all(imu.load_csv(path), activity, expected)
    width = max(len(r.name) for r in results)
    for r in results:
        print(f"  [{r.status}] {r.name:<{width}}  {r.detail}")
    return all(r.ok for r in results)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("activity", choices=config.RECORD_ACTIVITIES)
    ap.add_argument("--seconds", type=float, default=config.DEFAULT_RECORD_SECONDS)
    ap.add_argument("--countdown", type=float, default=config.DEFAULT_COUNTDOWN_SECONDS)
    ap.add_argument("--note", default="", help="free text saved in data/sessions.csv")
    ap.add_argument("--no-wait", action="store_true", help="queue the request and exit")
    args = ap.parse_args()

    if (config.CONTROL_DIR / recording.REQUEST_NAME).exists():
        print("A request is already waiting in data/control/request.json; delete it first if it's stale.")
        return 2
    req = recording.make_request(args.activity, args.seconds, args.countdown, args.note)
    recording.write_request(config.CONTROL_DIR, req)
    print(f"Requested: {args.activity}, {args.seconds:.0f} s after a {args.countdown:.0f} s countdown.")
    if args.no_wait:
        return 0

    try:
        st = wait_for(req)
    except KeyboardInterrupt:
        print("\nStopped watching. The App keeps recording; check LED4 or data/control/status.json.")
        return 0
    if st is None:
        print("The request stays queued and starts as soon as the App runs.")
        return 1
    if st["state"] != "done":
        print(f"App reported {st['state']}: {st.get('error', '')}")
        return 1
    return 0 if print_validation(st, args.activity) else 1


if __name__ == "__main__":
    sys.exit(main())
