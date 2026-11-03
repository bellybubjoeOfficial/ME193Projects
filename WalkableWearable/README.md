# Walkable Wearable — shin-mounted IMU step counter

The UNO Q on the shin streams IMU data over MQTT; its code is in
[ArduinoUnoQ/](ArduinoUnoQ/). The laptop does everything else (this folder):
acquisition, webcam, MediaPipe labeling, PyTorch training, live UI.
Built as standalone stages, each with its own test; a stage starts only after
the previous one passes. Every tunable value is in [config.yaml](config.yaml).

| Stage | What | Status |
|---|---|---|
| 1 | MQTT acquisition + clock sync + live plots | **45 s on the real UNO Q PASS — needs 10-min + disconnect runs** |
| 2 | Webcam acquisition (timestamped frames) | **built — needs 5-min run** |
| 3 | Vision-only heel-strike detector (MediaPipe) | **built — first real clips FAIL (see Stage 3)** |
| 4 | Synced recording + stomp alignment check | not started |
| 4 | Synced IMU + video recording, stomp alignment | **PASS on 2 real sessions (1–2 min); 5-min run pending** |
| 6 | IMU heel-strike CNN (labels from vision / hand) | **built — needs more sessions** |
| 7 | Live IMU-only step detection (holographic HUD) | **built — replay PASS; needs a live run on the shin** |
| 8 | Distance (step length × steps) | **built — ~9% held-out session error; needs a measured straight walk to confirm** |
| 5 | Dataset protocol (long passes, auto-split) | plan below |

## Setup (once)

```bash
# Python 3.14 is fine: mediapipe 0.10.35 supports it (Tasks API; mp.solutions is gone)
python3.14 -m venv ~/.venvs/me193walk
~/.venvs/me193walk/bin/python -m pip install -r requirements.txt
brew install mosquitto
```

The venv lives in `~/.venvs` because this folder's path contains a `:`, which
`venv` refuses.

Every command below runs **from this folder** with the venv's Python:

```bash
cd "ME193WalkWearable"
PY=~/.venvs/me193walk/bin/python
```

Start the broker in its own terminal (it listens on all interfaces so the UNO Q
can reach the laptop; point the Q at this laptop's IP, `ipconfig getifaddr en0`):

```bash
/opt/homebrew/sbin/mosquitto -c mosquitto.conf -v
```

## Message contract (the UNO Q publishes exactly this)

`wearable/imu/raw`
```json
{"device": "unoq-shin", "session": "<uuid>", "rate_hz": 100,
 "fields": ["ax","ay","az","gx","gy","gz"],
 "units": {"accel": "m/s^2", "gyro": "deg/s"},
 "samples": [[seq, t_mono_ns, t_wall_ns, ax, ay, az, gx, gy, gz], ...]}
```
- `seq`, `t_mono_ns`, `t_wall_ns` must be JSON **integers**; `fields` must be in exactly this order.
- `seq` increases by 1 per sample for the whole session (it never resets on Wi-Fi reconnect).
  A new `session` uuid = a new log file and fresh gap counters.

`wearable/status`: `{"device","online","session","rate_hz_measured","samples_sent",
"samples_buffered","reconnects","accel_range_g","gyro_range_dps","uptime_s"}` —
only `device` and `online` (bool) are required.

Clock sync: laptop publishes `wearable/sync/ping` `{"id","t_laptop_ns"}`; the device
replies on `wearable/sync/pong` `{"id","t_laptop_ns","t_dev_recv_ns","t_dev_send_ns"}`.
The pong timestamps must share a clock with one of the sample timestamps; `sync.device_clock`
says which. **The real UNO Q stamps pongs with its wall clock**, so samples convert via
`t_wall_ns` (`device_clock: wall`).

```
offset = ((t1 − t0) + (t2 − t3)) / 2    device − laptop
rtt    = (t3 − t0) − (t2 − t1)
```
Running median over the last 31 pings (1 per 2 s); pongs with rtt > 250 ms are logged
but ignored. Device time → laptop time: `t_laptop = t_mono − offset`.

Commands: `wearable/cmd` `{"cmd": "start"|"stop"|"set_rate", "value": ...}`.

**Laptop clock**: everything is stamped with `time.monotonic_ns()` ([laptop/clock.py](laptop/clock.py)),
which never jumps and is shared by all processes on the Mac — Stage 2's webcam uses the same one.

## Layout

```
laptop/acquire/   messages.py (parsers) seqtrack.py (gaps) ringbuffer.py logger.py acquirer.py
laptop/sync/      clocksync.py (NTP offset, running median, drift)
laptop/ui/        imu_live.py (6 plots + stats)
laptop/data/      load.py (read session folders back)
laptop/vision/    capture.py (Stage 2) frametiming.py | pose.py steps.py overlay.py
                  step_viewer.py handcheck.py compare.py clip.py (Stage 3)
laptop/train/     later stages
tests/            test_*.py (pytest, no hardware)  hw_*.py (hardware smoke tests)
                  fake_unoq.py (simulated UNO Q with fault injection)
data/sessions/    one folder per device session (git-ignored)
```

Session folder: `imu.parquet` (every sample in arrival order, raw device timestamps,
arrival time, offset at arrival, converted `t_laptop_ns`), `sync_log.csv`,
`status.jsonl`, `gaps.csv`, `meta.json`. Read with `laptop.data.load.load_imu(folder)`
(sorts by seq and keeps ns precision).

---

## Stage 1 — MQTT acquisition

**Unit tests** (no hardware):
```bash
$PY -m pytest -q
```
Covers parsing (types, precision of 1e18-ns timestamps, malformed payloads), gap detection,
out-of-order / late / duplicate samples, NTP offset math (symmetric, asymmetric, spikes,
rtt rejection, drift), ring buffer wrap and thread safety, log round-trip.

**Without the board** (proves the laptop side end to end):
```bash
$PY tests/fake_unoq.py                                     # terminal 2
$PY -m laptop.ui.imu_live                                  # terminal 3
$PY tests/fake_unoq.py --outage-at 20 --outage-for 3       # → one 3.00 s gap
$PY tests/fake_unoq.py --reorder 0.02                      # → late samples, no net loss
```

**Hardware pass test** (UNO Q streaming):
```bash
$PY tests/hw_stage1_imu.py                         # 10-minute clean run
$PY tests/hw_stage1_imu.py --minutes 3 --disconnect  # unplug/kill Wi-Fi on the Q mid-run, note how long
```
What you should see: six scrolling plots (accel left, gyro right, last 10 s); the line
breaks wherever samples are missing. The stats panel shows rate (last 5 s and whole run),
received/expected, missing/late/dup counts, gaps, latency (arrival − source time,
median/p95), clock offset, rtt, drift (after 2 min), and the device's status message.
Every gap is also printed in the terminal as it happens.

PASS (printed at the end, exit code 0):
- clean run: ≥ 99.9 % received, whole-run rate within 1 % of nominal (seq span ÷ laptop
  time span, so a mis-clocked device shows up), no gaps, clock synced, no malformed
  messages, one session, median UI refresh ≥ 24 fps
- `--disconnect`: ≥ 1 gap, every missing sample accounted for by a reported gap;
  printed gap durations should match how long you disconnected (if the Q buffers
  during the outage, those samples show as "recovered late" instead).

---

## Stage 2 — webcam acquisition

A capture thread stamps every frame with `now_ns()` (the Stage 1 clock) the moment
`read()` returns; a writer thread saves `video.mp4` and `frames.csv (frame_idx, t_ns)`
row-for-row, so the two always agree.

```bash
$PY -m laptop.vision.capture --probe                 # what modes the camera really delivers
$PY -m laptop.vision.capture --label test --minutes 1  # record a clip (q stops)
$PY tests/hw_stage2_webcam.py                        # 5-minute pass test
$PY tests/hw_stage2_webcam.py --check data/clips/<folder>   # re-check a saved clip
```
What you should see: the preview with resolution, current/run fps, dropped frames, and
REC. PASS: every 10 s window within 5 % of the run mean fps, 0 dropped frames,
frames.csv rows == decoded video frames == captured frames, frame_idx consecutive.

Built-in camera (probed 2026-11-02): ~29.9 fps at 1280×720 and 1920×1080; 60 fps is not
available; 640×480 only gave 23 fps. Low light makes macOS lengthen exposure and the
fps drops — the test reports that as low fps, not as drops.

**Drops vs late stamps**: a long interval followed by a short one (e.g. 57 ms + 8 ms)
is a frame that arrived on time but was stamped late (the thread was briefly blocked).
These are not counted as drops but are reported ("stamped late, worst N ms") because
they limit Stage 4's frame-level alignment.

## Stage 3 — vision-only step detector

MediaPipe `PoseLandmarker` (Tasks API; models download to `models/` on first run).
Clips use the **heavy** model (finds a legs-only walker in ~90 % of frames vs ~50 % for
`full`; ~19 fps offline); the live view uses `full` for speed. Landmarks are cached per
model in `landmarks_<model>.parquet`.

Detection method (`steps.method`, all distances in leg lengths so camera distance doesn't
matter) — see [steps.py](laptop/vision/steps.py):
- **`footprint` (default)**: a heel strike is a heel arriving at a new spot on the floor and
  staying there for `min_stance_frames`. Both heels are treated as unlabeled points, because
  MediaPipe **swaps left/right** when the legs cross in a side view (seen constantly in the
  2026-11-03 clips). L/R is assigned afterwards from the majority label (alternating when unclear).
- `heel_x`: per-leg heel stops sliding forward (needs clean L/R labels).
- `heel_y`: per-leg heel stops coming down. Fails with a floor-level camera: the big heel
  rise on screen is the heel peeling up *before* toe-off, not the swing.

First real results (2026-11-03, floor-level camera, heavy model, footprint method):

| clip | true | detected | missed | extra | median timing | L/R right |
|---|---|---|---|---|---|---|
| slow_walk | 20 | 25 | 11 | 16 | 2 frames | 75 % |
| normal_walk | 31 | 27 | 6 | 2 | 4 frames (det. ~3 early) | 96 % |
| jog | 30 | 26 | 5 | 1 | 3 frames (det. ~3 early) | 100 % |

Open issues: (1) misses cluster where the walker enters the frame / turns around and where
pose is lost; (2) detections are a consistent ~3 frames before the hand marks — the heel
stops sliding before it visibly touches down, so "strike" needs one agreed definition;
(3) slow_walk's marks have long gaps (286–349, 516–613, 684–781) where the walker kept
walking — most of its "extras" fall there and need re-checking.

**Unit tests** (synthetic trajectories, [tests/synth_gait.py](tests/synth_gait.py)): footprint method with
and without 30 % random L/R label swaps, standing still; heel_y method on slow walk / walk / jog × hard / soft landing: exact counts, median timing error
≤ 1 frame (hard landings: 0); plus noise up to 2 % of leg length, standing still,
low visibility, short dropouts, 2 s occlusion, heel bounce, 15 fps camera, both legs at
two camera distances.

**Workflow** — camera side-on, whole legs in frame, good light, ~20 steps per clip:
```bash
$PY -m laptop.vision.step_viewer --camera                 # live: skeleton, L/R flash, counts
$PY -m laptop.vision.capture --label slow_walk --minutes 0.5   # record 3 clips:
$PY -m laptop.vision.capture --label normal_walk --minutes 0.5 #   slow, normal, jog
$PY -m laptop.vision.capture --label jog --minutes 0.5
$PY -m laptop.vision.step_viewer --clip data/clips/<folder>    # detect + watch (--save-video)
$PY -m laptop.vision.handcheck --clip data/clips/<folder>      # mark the true strikes
$PY tests/hw_stage3_steps.py data/clips/*_slow_walk data/clips/*_normal_walk data/clips/*_jog
```
`--camera --record LABEL` records a clip while showing live detection. Phone videos
work too: `--clip video.mov` (timestamps from the file's fps; outputs in `video_steps/`).
Landmarks are cached in `landmarks.parquet`, so after changing a `steps:` threshold the
re-run is instant (`--repose` redoes pose).

Hand-check keys: `d`/`a` (or arrows) ±1 frame, `D`/`A` ±10, space play, `n`/`p` jump to
next/prev detection, `l`/`r` mark a left/right strike (toggle), `x` delete marks nearby,
`q` save + quit + report. Mark the **first frame the heel touches the ground**.
The timeline shows detections (colored, top) and your marks (white, bottom).

PASS per clip: total count within ±1 of your marks, median |timing error| ≤ 2 frames.
The report explains each miss/extra from landmark visibility (e.g. "R heel below
visibility 0.5 in 34 % of frames (far leg occluded?)"). Expected weak spots: the far
leg hidden behind the near leg, motion blur when jogging in dim light, and the frames
where you turn around (stop/turn at the end of the course, or trim it out).

---

## Stage 4 — synced recording + stomp check

One command records IMU (Stage 1) and webcam (Stage 2) into one session folder,
`data/recordings/<time>_<speed>/`: `imu.parquet sync_log.csv status.jsonl gaps.csv imu_meta.json`,
`video.mp4 frames.csv camera.json`, `meta.json` (course length, speed, IMU leg, notes, run
summary), then the stomp check writes `stomp.json stomp.png sync_correction.json`.

```bash
$PY -m laptop.sync.record --speed normal --course 10 --leg R --notes "tape course"
$PY tests/hw_stage4_sync.py --course 10 --leg R          # 5-minute pass session
$PY tests/hw_stage4_sync.py --check data/recordings/<folder>
$PY -m laptop.sync.stomp data/recordings/<folder>        # stomp check only
```
It waits for IMU data + clock sync before the camera starts. The preview shows a live
`|accel| − 1 g` trace with the stomp threshold, IMU/video health, and the protocol:
**stand still → STOMP the IMU foot (feet visible to the camera) → walk → stand still →
STOMP → q**.

Stomp check ([stomp.py](laptop/sync/stomp.py)): IMU impact = where `|a| − 1 g` first crosses
half the spike's peak (interpolated, ≥ 3 g to count); video impact = first frame the higher
heel on screen is back on the ground after lifting (label-agnostic, heavy pose model only
around the IMU stomp). residual = video − IMU on the laptop clock; drift = end − start.
`sync_correction.json` maps IMU time to video time (linear between stomps) for later stages.
PASS: both stomps found in both streams, |residual| ≤ 1 frame, drift ≤ 10 ms.

Expect a positive residual: frames are stamped when OpenCV hands them over, after the
camera's own capture latency.

## Stage 6 — IMU step model

```bash
$PY -m laptop.train.train              # every session in data/recordings with a stomp sync
$PY -m laptop.train.train --sessions data/recordings/A data/recordings/B --iterations 500
```
- **Labels** ([labels.py](laptop/train/labels.py)): `truth.csv` in the session folder if you hand-marked it
  (`handcheck --clip data/recordings/<folder>`), else the vision detector on the session video
  (heavy pose, cached). Moved onto the IMU clock with `sync_correction.json`.
- **Data** ([dataset.py](laptop/train/dataset.py)): IMU on a seq-indexed 100 Hz grid; targets = a Gaussian
  bump (30 ms) per strike, one channel per leg. Only *labeled* samples count — between the
  stomps (±1.5 s margin) and where the camera saw a heel — so "not visible" never teaches
  "no step".
- **Model** ([model.py](laptop/train/model.py)): fully convolutional 1D CNN, 6 → 2 channels, dilated
  residual blocks, 1.27 s receptive field. Masked, positive-weighted BCE; random 2.56 s windows
  with gain/noise augmentation.
- **Evaluation**: leave-one-session-out, peaks above 0.5 with a 0.3 s refractory per leg, matched
  within 67 ms (2 frames): precision / recall / F1 / timing, next to a no-learning baseline
  (accel-magnitude peaks). Final model on all sessions → `models/step_cnn.pt`, report
  `models/step_cnn_report.json`, plot `models/step_cnn_eval.png`.
- Scores measure agreement **with the labels**; vision labels are only as good as Stage 3.

Sessions skipped automatically: notes containing "smoke", or no stomp sync.

## Stage 7 — live mode

```bash
$PY -m laptop.ui.live_steps                                   # live from the UNO Q
$PY -m laptop.ui.live_steps --replay data/recordings/<folder> # replay a session, no hardware
$PY tests/hw_stage7_live.py --replay data/recordings/<folder> # live vs offline, delay
$PY tests/hw_stage7_live.py --ui data/recordings/<folder>     # HUD frame rate
$PY tests/hw_stage7_live.py --live-log data/live/<time>_steps.csv --session data/recordings/<folder>
```
[stream.py](laptop/train/stream.py) runs the Stage 6 CNN over the last 4 s of IMU 15×/s and calls a
step once `live.lookahead_s` (0.3 s) of IMU after it has arrived. Exact certainty would need half
the 1.27 s receptive field (0.64 s); on both 2026-11-03 sessions 0.3 s kept every step within
30 ms of offline at a **~0.39 s median delay** (vs ~0.73 s). Steps are tracked by IMU seq, so
overlapping updates never double count; a device restart resets the counters.

HUD ([live_steps.py](laptop/ui/live_steps.py), theme in [theme.py](laptop/ui/theme.py)):
- **LIVE** — step-counter ring (shockwave flash: cyan = left, magenta = right), L / R
  counts, cadence (median step interval), last step (probability + delay); strike
  probability per leg (solid = final, dotted = provisional, violet FINAL|LIVE boundary,
  amber threshold, ▲ on each counted step); raw |accel| and |gyro|; event log; status chips
  (link, sync, IMU rate, loss, delay, UI fps).
- **PARAMETERS** — live-tunable, applied instantly, SAVE writes `config.yaml` keeping comments:
  threshold, refractory, lookahead, flash, plot window. Retrain-required (read from the model
  file): window, label width, CNN width, receptive field, iterations, augmentation…

Every counted step is saved on close to `data/live/<time>_steps.csv` (laptop clock), so a live
run during a Stage 4 recording can be scored against that recording's labels (`--live-log`).

## Stage 8 — distance

```bash
$PY -m laptop.train.distance_eval     # compare methods, save models/distance.json + figure
```
[distance.py](laptop/train/distance.py): distance = Σ step lengths. Methods, each fitted on 7 sessions
and tested on the 8th: **prior** (Apple 0.63 m), **fixed** (one calibrated length), **cadence**
(a + b × step frequency), **pendulum** (shin swing angle from the gyro × leg length).

Ground truth = hip travel on straight stretches of the session video, with **your leg as the
ruler** (straight hip→ankle = 1.016 − 0.07 m). The floor tape (85 in apart) was tried first but
sits nearer the camera than the feet and overstated the scale ~30%.

First results (2026-11-03, 8 sessions, steps from the final step model = in-sample):

| method | per pass |err| (95% CI) | per session |err| | bias |
|---|---|---|---|
| prior 0.63 m | 14.7% (12.7–16.6) | 12.4% | −10% |
| **fixed 0.70 m** | 14.1% (11.8–16.6) | **9.3%** | 0% |
| cadence | 14.3% (12.0–16.7) | 9.4% | −1% |
| pendulum | 25.3% (22.4–28.2) | 22.0% | −8% |
| walk ratio (Apple 0.0055) | 31.0% (28.6–33.5) | 31.1% | −31% |
| walk ratio (fitted 0.0078) | 16.3% (13.9–19.0) | 8.4% | −3% |
| swing speed (peak shin gyro) | 14.5% (12.0–17.2) | 9.3% | +1% |

Speed-dependent step length (walk ratio, swing speed) doesn't beat a fixed length **on this data**:
per-step cadence here is dominated by turning on a 3 m course, so it doesn't reflect walking
speed. Live mode uses **fixed** unless another method is ≥ 2 points better per session and no
worse per pass. Long straight walks at several speeds are what these methods need.

Limits: passes are only 2–4 steps (a 3 m course with turnarounds), so cadence barely varies and
±1 step is ±30% of a pass; the video ruler may itself read ~10% long (Apple says 0.63 m/step).
Re-run `laptop.train.train` to save per-fold step models (`models/folds/`) so distance uses
step counts from a model that never saw the session. The live HUD shows **DISTANCE ± band**
from `models/distance.json`.

## Later stages (plan)

**5 Dataset** — red-tape course, multiple speeds, auto-split passes, turnarounds labeled.
**6 Training** — 1D CNN on IMU windows, MediaPipe heel strikes (both legs) as labels,
held-out-session evaluation. **7 Live mode** — IMU-only inference, step flash, probability
plot, parameters page (live-tunable vs retrain-required). **8 Distance** — per-step stride
calibrated on course length, uncertainty analysis, writeup figures.
