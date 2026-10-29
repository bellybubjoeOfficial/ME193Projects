#!/usr/bin/env python3
"""Stages 6-7: live activity recognition, shown on the LED matrix and in this terminal.

Run ON THE Q (over SSH), with the activity_wearable App running and
models/ trained (scripts/train.py):

    python scripts/live.py                  # Model 2: Random Forest, all IMU outputs
    python scripts/live.py --model 1        # Model 1: peak acceleration on one axis
    python scripts/live.py --detach         # start live mode and exit (keeps running untethered)
    python scripts/live.py --stop           # end live mode

ENTER here ends live mode. Ctrl-C (or a dropped SSH connection) only closes
this viewer: the App keeps predicting and showing it on the matrix.

The App predicts every 1.28 s on the last 2.56 s and shows a majority vote
of the last 3 predictions, so a change of activity takes ~3-5 s to appear.

Offline (laptop or Q, no App needed): run a recorded file through the same
live code and see what it would have shown:

    python scripts/live.py --replay data/raw/ankle_walk_s02_2026-10-27T14-19.csv --model 1
"""

import argparse
import select
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wearable import config, display, forest_lite, imu, live, recording  # noqa: E402

PICKUP_TIMEOUT_S = 5


def model_path(number):
    return config.MODELS_DIR / config.LIVE_MODEL_FILES[f"model{number}"]


def describe(model):
    m = model.get("meta", {})
    extra = f", axis {m['axis']}, depth {m['depth']}" if "axis" in m else ""
    return (f"{model['kind']}: {len(model['trees'])} tree(s), {len(model['features'])} feature(s){extra}; "
            f"trained {m.get('trained_at', '?')}, CV macro-F1 {m.get('cv_macro_f1', '?')}")


def line(p):
    probs = "  ".join(f"{c} {v:.2f}" for c, v in p["proba"].items())
    return f"{p['label'].upper():6s} (this window: {p['raw_label']:5s}, vote {p['vote_fraction']:.0%})   {probs}"


def replay(path, number):
    model = forest_lite.load(model_path(number))
    df = imu.load_csv(path)
    print(f"Replaying {path} through {describe(model)}\n")
    preds = live.replay(model, df)
    step_s = config.WINDOW_SAMPLES * (1 - config.WINDOW_OVERLAP) / config.SAMPLE_RATE_HZ
    first_s = config.WINDOW_SAMPLES / config.SAMPLE_RATE_HZ
    for i, p in enumerate(preds):
        print(f"  t={first_s + i * step_s:6.2f} s  {line(p)}")
    try:
        truth = config.ACTIVITY_TO_CLASS.get(recording.parse_filename(path)[0])
    except ValueError:
        truth = None
    if truth and preds:
        ok = sum(p["label"] == truth for p in preds)
        print(f"\n{ok}/{len(preds)} shown predictions = '{truth}' (the file's label). "
              "If this file was used for training, this is optimistic.")
    return 0


def enter_pressed(timeout):
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    return bool(ready) and sys.stdin.readline() != ""


def watch(show_matrix=False):
    """Print every new prediction from status.json until ENTER (stop) or Ctrl-C (leave running)."""
    last_n = None
    print("ENTER = stop live mode, Ctrl-C = close this viewer only\n")
    try:
        while True:
            if enter_pressed(0.3):
                recording.write_stop(config.CONTROL_DIR)
                print("Live mode stopped.")
                return 0
            st = recording.read_status(config.CONTROL_DIR) or {}
            if st.get("state") != "live":
                print(f"\nThe App left live mode (state: {st.get('state')}"
                      + (f", {st['error']}" if st.get("error") else "") + ").")
                return 1
            if st.get("label") and st.get("n") != last_n:
                last_n = st["n"]
                print(f"  {time.strftime('%H:%M:%S')}  {line(st)}", flush=True)
                if show_matrix:
                    frame = display.activity(st["label"], st["vote_fraction"], 1 if st["model"] == "model1" else 2)
                    print(display.as_text(frame) + "\n", flush=True)
    except KeyboardInterrupt:
        print("\nViewer closed; the App is still predicting (LED matrix). `python scripts/live.py --stop` ends it.")
        return 0


def start(number):
    path = model_path(number)
    if not path.exists():
        print(f"{path} doesn't exist yet: run python scripts/train.py first.")
        return 2
    print(describe(forest_lite.load(path)))
    recording.write_live_request(config.CONTROL_DIR, f"model{number}")
    t0 = time.time()
    while (config.CONTROL_DIR / recording.LIVE_NAME).exists():
        if time.time() - t0 > PICKUP_TIMEOUT_S:
            print("The App hasn't picked up the request. Is it running, and restarted since Stage 6 was added?\n"
                  "  arduino-app-cli app list\n  arduino-app-cli app restart ~/ArduinoApps/activity_wearable")
            (config.CONTROL_DIR / recording.LIVE_NAME).unlink(missing_ok=True)
            return 1
        time.sleep(0.3)
    st = recording.read_status(config.CONTROL_DIR) or {}
    if st.get("state") != "live":
        print(f"The App didn't start live mode: {st.get('error', st)}")
        return 1
    print(f"Live mode on (model {number}). First prediction in ~2.6 s.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=int, choices=[1, 2], default=2, help="1 = one-axis peak, 2 = Random Forest")
    ap.add_argument("--detach", action="store_true", help="start live mode and exit")
    ap.add_argument("--stop", action="store_true", help="end live mode")
    ap.add_argument("--replay", type=Path, help="run a recorded CSV through the live code instead (offline)")
    ap.add_argument("--matrix", action="store_true", help="also draw the LED-matrix picture in the terminal")
    args = ap.parse_args()
    if args.replay:
        return replay(args.replay, args.model)
    if args.stop:
        recording.write_stop(config.CONTROL_DIR)
        print("Asked the App to end live mode.")
        return 0
    rc = start(args.model)
    if rc or args.detach:
        return rc
    return watch(args.matrix)

if __name__ == "__main__":
    sys.exit(main())
