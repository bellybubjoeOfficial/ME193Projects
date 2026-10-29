"""activity_wearable — Linux side. Runs inside the App Lab container.

Stage 1 (connection check), on every start:
  * asks the sketch what it found on the I2C buses (Bridge.call("imu_info"))
    and saves it to data/check/imu_info.json
  * receives every 100 Hz sample (Bridge.notify("imu", ...) from the sketch),
    converts counts -> g and °/s, and records the first CHECK_SECONDS to
    data/check/imu_check.csv
  * logs a once-per-second status line (see: arduino-app-cli app logs … --follow)
Then run scripts/check_imu.py to get PASS/FAIL for each check.

Stages 3-4 (recordings):
  * polls data/control/request.json (written by scripts/collect.py or record.py)
  * counts down, records data/raw/ankle_<activity>_sNN_<time>.csv, appends a
    row to data/sessions.csv, and keeps data/control/status.json up to date
  * data/control/stop.json (ENTER in collect.py) cancels the countdown or ends
    the recording early, keeping what was recorded
  * shows the state on RGB LED4 and the LED matrix (wearable/display.py):
      idle       LED off,    matrix: one dim dot (bottom right)
      countdown  LED blue,   matrix: seconds left
      recording  LED red,    matrix: seconds left + progress bar
      done       LED green,  matrix: check mark
      error      LED purple, matrix: X
The recording runs inside the App, not in your SSH session, so it keeps
going if SSH drops or the laptop goes away.

Stages 6-7 (live inference):
  * scripts/live.py writes data/control/live.json {"model": "model1"|"model2"};
    the App loads models/<model>.json (plain JSON trees, numpy only: the
    container has no scikit-learn) and predicts every 1.28 s
  * shows the smoothed prediction on the LED matrix (SIT/STD/WLK/RUN/AIR +
    confidence bar; 1 or 2 dots top-right = which model), LED4 green, and
    writes it to data/control/status.json (state "live") for live.py to print
  * data/control/stop.json, or a new recording request, ends live mode
  * when scripts/train.py saves a new model, live mode switches to it by itself

The App folder is bind-mounted at /app in the container, so files written
under /app/data land in ~/ArduinoApps/activity_wearable/data on the board.
"""

import json
import math
import queue
import sys
import time
from pathlib import Path

# The `wearable` package lives in the project root, one level above python/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arduino.app_utils import App, Bridge, Logger

from wearable import config, display, forest_lite, imu, live, recording

logger = Logger("activity_wearable")

# LED4 colours, must match set_status() in the sketch.
LED_OFF, LED_COUNTDOWN, LED_RECORDING, LED_DONE, LED_ERROR = range(5)

capture = recording.CsvRecorder(config.CHECK_DIR / "imu_check.csv", config.CHECK_SECONDS)
unwrap_t = imu.Unwrapper()
unwrap_seq = imu.Unwrapper()
stats = {"n": 0, "last": None}
info = None

# The recording in progress. `rec` is only set while samples should be
# written; the Bridge thread reads it, the main loop sets/clears it.
job = None          # dict: request + file/session/start info
rec = None          # recording.CsvRecorder
countdown_end = None
shown = {"led": None, "frame": None}

# Live inference (Stages 6-7). The Bridge thread only queues samples; the
# main loop does the (slower) feature + model work, so it never delays samples.
predictor = None    # live.LivePredictor while live mode is on
live_model = None   # "model1" | "model2"
live_mtime = None   # model file time when loaded: a newer file (train.py re-run) is reloaded
live_q = queue.SimpleQueue()


def on_imu(seq, t_us, ax, ay, az, gx, gy, gz):
    """Bridge.notify("imu", …) from the sketch: one sample, raw counts."""
    row = imu.sample_row(unwrap_seq(seq), unwrap_t(t_us), time.time(), (ax, ay, az, gx, gy, gz))
    stats["n"] += 1
    stats["last"] = row
    if capture.add(row) and not stats.get("check_logged"):
        stats["check_logged"] = True
        logger.info(f"Check capture complete: {capture.path}  -> now run scripts/check_imu.py")
    r = rec
    if r is not None:
        r.add(row)
    if predictor is not None:
        live_q.put(row)


Bridge.provide("imu", on_imu)


def show(led, frame):
    """Update LED4 and the matrix, only sending what changed."""
    # The LED and matrix are only a convenience; never stop a recording over them.
    if led != shown["led"]:
        try:
            Bridge.notify("set_status", led)
            shown["led"] = led
        except Exception as e:
            logger.warning(f"set_status failed: {e}")
    if frame != shown["frame"]:
        try:
            Bridge.call("draw", display.to_bytes(frame), timeout=1)
            shown["frame"] = frame
        except Exception as e:
            logger.warning(f"matrix draw failed: {e}")


def fetch_info():
    """Ask the sketch for its scan/config result; save it; return the parsed dict."""
    try:
        text = Bridge.call("imu_info", timeout=2)
    except Exception as e:  # MCU still booting, or sketch not flashed yet
        logger.warning(f"imu_info not available yet: {e}")
        return None
    parsed = imu.parse_info(text)
    config.CHECK_DIR.mkdir(parents=True, exist_ok=True)
    (config.CHECK_DIR / "imu_info.json").write_text(
        json.dumps({**parsed, "saved_at": time.time()}, indent=2))
    return parsed


# ---- Recording state machine (called once per loop, ~1 s) -------------------

def poll_request():
    """idle -> countdown when a script has left a request."""
    global job, countdown_end
    try:
        req = recording.take_request(config.CONTROL_DIR)
    except ValueError as e:
        logger.warning(str(e))
        recording.write_status(config.CONTROL_DIR, "error", error=str(e))
        show(LED_ERROR, display.error())
        return
    if req is None:
        return
    job = req
    countdown_end = time.time() + req["countdown"]
    logger.info(f"Request: {req['activity']} for {req['seconds']:.0f} s, starting in {req['countdown']:.0f} s")


def start_recording():
    """countdown -> recording: pick the file name and start writing samples."""
    global rec
    session = recording.next_session_number(config.RAW_DIR, job["activity"])
    started = time.time()
    path = config.RAW_DIR / recording.recording_filename(job["activity"], session, started)
    job.update(session=session, file=path.name, started_at=started)
    r = recording.CsvRecorder(path, job["seconds"])
    r.arm()
    rec = r
    logger.info(f"Recording {path.name} ({job['seconds']:.0f} s)")


def finish_recording():
    """recording -> done: log the session and go back to waiting."""
    global job, rec
    r = rec
    rec = None
    recording.append_session(config.SESSIONS_CSV, {
        "file": job["file"], "activity": job["activity"], "session": job["session"],
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(job["started_at"])),
        "requested_s": job["seconds"], "seconds": round(r.elapsed_s(), 2), "samples": r.n,
        "dropped": r.dropped, "stopped_early": int(r.stopped_early), "note": job["note"],
    })
    recording.write_status(config.CONTROL_DIR, "done", activity=job["activity"], file=job["file"],
                           samples=r.n, dropped=r.dropped, seconds=round(r.elapsed_s(), 2),
                           stopped_early=r.stopped_early)
    show(LED_DONE, display.done())
    how = "stopped early" if r.stopped_early else "complete"
    logger.info(f"Recording {how}: {job['file']}  {r.n} samples, {r.dropped} dropped")
    job = None


def cancel_countdown():
    global job, countdown_end
    logger.info(f"Cancelled {job['activity']} during the countdown; nothing recorded")
    recording.write_status(config.CONTROL_DIR, "cancelled", activity=job["activity"])
    show(LED_OFF, display.idle())
    job = countdown_end = None


def step_recording():
    global countdown_end
    stop = recording.take_stop(config.CONTROL_DIR)
    if job is None:
        poll_request()                       # a stop with nothing running is just discarded
        if job is None:
            return
    if rec is None:                          # counting down
        if stop:
            cancel_countdown()
            return
        left = countdown_end - time.time()
        if left <= 0:
            countdown_end = None
            start_recording()
        else:
            recording.write_status(config.CONTROL_DIR, "countdown", activity=job["activity"],
                                   remaining_s=round(left, 1))
            show(LED_COUNTDOWN, display.countdown(math.ceil(left)))
            logger.info(f"Starting {job['activity']} in {left:.0f} s")
            return
    if stop:
        rec.stop()
    if rec.done:
        finish_recording()
    else:
        elapsed = rec.elapsed_s()
        recording.write_status(config.CONTROL_DIR, "recording", activity=job["activity"], file=job["file"],
                               elapsed_s=round(elapsed, 1), seconds=job["seconds"], dropped=rec.dropped)
        show(LED_RECORDING, display.recording(math.ceil(job["seconds"] - elapsed), elapsed / job["seconds"]))


# ---- Live inference (called once per loop, ~1 s) ------------------------------

def drain_live_queue():
    rows = []
    while True:
        try:
            rows.append(live_q.get_nowait())
        except queue.Empty:
            return rows


def start_live(name):
    global predictor, live_model, live_mtime
    path = config.MODELS_DIR / config.LIVE_MODEL_FILES[name]
    try:
        mtime = path.stat().st_mtime
        model = forest_lite.load(path)
    except (OSError, ValueError) as e:
        logger.warning(f"can't load {path}: {e}  (run scripts/train.py first)")
        recording.write_status(config.CONTROL_DIR, "error", error=f"can't load {path.name}: {e}")
        show(LED_ERROR, display.error())
        return
    drain_live_queue()
    predictor, live_model, live_mtime = live.LivePredictor(model), name, mtime
    meta = model.get("meta", {})
    logger.info(f"Live inference with {name}: {len(model['features'])} features, "
                f"{len(model['trees'])} tree(s), trained {meta.get('trained_at', '?')}")
    recording.write_status(config.CONTROL_DIR, "live", model=name, label=None, n=0,
                           trained_at=meta.get("trained_at"))
    show(LED_DONE, display.idle())


def stop_live(why):
    global predictor, live_model
    logger.info(f"Live inference stopped ({why})")
    predictor = live_model = None
    drain_live_queue()
    recording.write_status(config.CONTROL_DIR, "idle")
    show(LED_OFF, display.idle())


def step_live():
    """Returns True while live mode owns the App (then the recording machine waits)."""
    if job is not None:                       # never interrupt a recording
        return False
    try:
        name = recording.take_live_request(config.CONTROL_DIR)
    except ValueError as e:
        logger.warning(str(e))
        name = None
    if name is not None:                      # start, or switch model
        start_live(name)
    if predictor is None:
        return False
    try:                                      # scripts/train.py saved a new model: switch to it
        if (config.MODELS_DIR / config.LIVE_MODEL_FILES[live_model]).stat().st_mtime != live_mtime:
            logger.info(f"{live_model} was retrained: reloading it")
            start_live(live_model)
            if predictor is None:
                return False
    except OSError:
        pass                                  # file briefly missing: keep the model we have
    if recording.take_stop(config.CONTROL_DIR):
        stop_live("stop requested")
        return False
    if (config.CONTROL_DIR / recording.REQUEST_NAME).exists():
        stop_live("recording requested")
        return False
    out = None
    for row in drain_live_queue():
        out = predictor.add(row) or out
    if out is not None:
        recording.write_status(config.CONTROL_DIR, "live", model=live_model, **out)
        show(LED_DONE, display.activity(out["label"], out["vote_fraction"], 1 if live_model == "model1" else 2))
        logger.info(f"[{live_model}] {out['label']:5s} (window: {out['raw_label']}, "
                    f"vote {out['vote_fraction']:.0%})")
    return True


def loop():
    global info
    if info is None:
        parsed = fetch_info()
        if parsed is None:
            time.sleep(1)
            return
        if not parsed["ok"]:
            logger.warning(f"IMU NOT READY: {parsed['raw']}")
            time.sleep(2)
            return
        info = parsed
        logger.info(f"IMU ready: {info['raw']}")
        capture.arm()
        logger.info(f"Recording {capture.seconds} s check capture — keep the board still")
        recording.take_stop(config.CONTROL_DIR)          # stale from before a restart
        recording.write_status(config.CONTROL_DIR, "idle")
        show(LED_OFF, display.idle())

    time.sleep(1)
    if not step_live():
        step_recording()
    n, stats["n"] = stats["n"], 0
    last = stats["last"]
    if last:
        a, g = last[3:6], last[6:9]
        mag = sum(v * v for v in a) ** 0.5
        logger.info(f"{n:3d} samples/s  a[g]={a} |a|={mag:.3f}  g[dps]={g}")
    else:
        logger.warning("no samples arriving from the sketch")


App.run(user_loop=loop)
