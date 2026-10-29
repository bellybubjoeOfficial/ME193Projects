# 🏃 activity_wearable

Ankle-worn IMU activity classifier on an Arduino UNO Q: standing, walking,
running, air-biking (pedalling while lying on your back) and sitting. Two classical-ML models, no
neural nets:

- **Model 2:** Random Forest on features from all six IMU channels.
- **Model 1:** peak acceleration along one axis per window, with a shallow tree / thresholds.

## How the pieces fit

```
MPU-6050 ──I2C──> STM32 MCU (sketch/sketch.ino)          100 Hz, raw int16 counts
                    │ Bridge.notify("imu", seq, t_us, ax..gz)
                    ▼
                  Linux side, App Lab container (python/main.py)
                    │ counts -> g, °/s ; writes CSV under data/
                    ▼
                  ~/ArduinoApps/activity_wearable/data/ on the Q   (scp to laptop)
```

On the UNO Q the header pins belong to the STM32 MCU, not Linux. So the MCU
reads the sensor, and Python receives the data over the Router Bridge. Python
runs inside a Docker container that App Lab starts. The App folder is mounted
at `/app`, so files the app writes land in this folder on the board.

## Project layout

| Folder | What | Why it differs from the original plan |
|---|---|---|
| `sketch/` | MCU firmware | Planned as `firmware/`. App Lab requires the name `sketch/`. |
| `python/main.py` | App entry point (Bridge receiver, logger) | App Lab requires `python/main.py`. |
| `wearable/` | Python package: IMU I/O, checks (features, models later) | Plain Python, no Arduino imports, so it runs and tests on the laptop too. |
| `scripts/` | Command-line tools (`check_imu.py`, …) | |
| `data/raw/`, `data/sessions.csv` | recordings + metadata (from Stage 3) | |
| `data/check/` | Stage 1 connection-check capture | |
| `models/` | trained models: `model1.json`, `model2.json` (live App), `model2.joblib` (laptop), `*.h` (optional C) | |
| `reports/` | Stage 5-7 report + plots (`latest/`), `history.csv` | |
| `tests/` | pytest, one file per stage | |

## Setup

**UNO Q (Linux host, for tests and scripts).** The board's Python has no pip, so we use `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh          # once; installs ~/.local/bin/uv
uv venv ~/.venvs/wearable
uv pip install -p ~/.venvs/wearable -r requirements-dev.txt
source ~/.venvs/wearable/bin/activate
```

**Laptop:** `python -m venv .venv && source .venv/bin/activate && pip install -r requirements-dev.txt`

The App itself needs nothing extra: the App Lab container already has numpy + pandas.

## Stage 1: IMU connection check

**Wiring** (board unpowered while you wire):

| IMU breakout | UNO Q |
|---|---|
| VCC | 5V **only if the breakout has an on-board 3.3 V regulator** (GY-521 does); otherwise 3.3V |
| GND | GND |
| SDA | **A4** |
| SCL | **A5** |

On the UNO Q, A4/A5 are **not** the same wires as the SDA/SCL header pins
(they are on a classic Uno R3). They're a separate I2C bus (`Wire2`). The
SDA/SCL header pins on this board are damaged, so the sketch only scans
`Wire2` (A4/A5) and `Wire1` (Qwiic) and reports where it found the IMU.
A4/A5 are **not 5 V tolerant**: the I2C pull-ups must go to 3.3 V (a GY-521's
on-board pull-ups do, even with VCC on 5V).

**Run** (on the Q, over SSH; this stops whatever app is running):

```bash
cd ~/ArduinoApps/activity_wearable
arduino-app-cli app start .            # compiles + flashes the sketch, starts python/main.py
arduino-app-cli app logs . --follow    # Python: "IMU ready …", then one line per second
arduino-app-cli monitor                # MCU: silent when OK; "IMU NOT READY: …" if the sensor isn't found
                                       # (set DEBUG_PRINT = true in the sketch for a per-second status line;
                                       #  it costs ~15-20 ms of sampling delay per print, so keep it off for recordings)
```

Leave the board **flat and untouched for 30 s** after "Recording 30 s check
capture" appears in the logs, until "Check capture complete" appears. Then:

```bash
python scripts/check_imu.py            # PASS/WARN/FAIL table + troubleshooting on failure
python -m pytest -v                    # unit tests + hardware tests on the real capture
```

What is checked: device found at 0x68/0x69, WHO_AM_I = 0x68, ±16 g / ±2000 °/s
read back from the sensor, all channels updating, |a| = 1 ± 0.1 g, gyro bias
under 5 °/s (WARN up to the ±20 °/s datasheet limit) with noise under 1 °/s,
100 Hz ±1 % from MCU timestamps, zero dropped samples (sequence numbers), and
the Linux side keeping up with the MCU.

Tests: `tests/test_stage1_imu.py` (synthetic, no hardware) and
`tests/test_stage1_hardware.py` (skipped until a real capture exists).

## Stage 2: features + models (offline, synthetic data)

No hardware and no App needed; this runs in the test venv (needs scipy +
scikit-learn from `requirements-dev.txt`).

- `wearable/features.py`: gravity/body split (0.3 Hz low-pass over the whole
  recording), 2.56 s windows with 50 % overlap, ~70 features per window.
- `wearable/models.py`: Random Forest (Model 2) plus a comparison zoo, scored
  with session-grouped cross-validation so overlapping windows from one
  recording never land in both train and test.
- `wearable/synth.py`: caricature recordings per activity, used by the tests
  until real data exists.

```bash
python -m pytest tests/test_stage2_features.py tests/test_stage2_models.py -v   # ~2 min on the Q
python -m pytest tests/test_stage2_features.py                                  # fast loop while editing
```

The accuracy thresholds in `test_stage2_models.py` use easy synthetic classes:
they prove the pipeline works, not how well it does on real recordings.

**Note for Stage 6:** the App Lab container has numpy + pandas only. Live
inference in `python/main.py` will need scipy + scikit-learn added via
`python/requirements.txt`.

## Stage 3: first real recording (sitting)

**Install:** `matplotlib` (now in `requirements-dev.txt`). On the Q:
`uv pip install -p ~/.venvs/wearable -r requirements-dev.txt`; on the laptop:
`pip install -r requirements-dev.txt`. Nothing new in the App container.

**How recording works.** The App (`python/main.py`, in its container) does
the recording; `scripts/record.py` (on the Q, over SSH) just leaves a request
in `data/control/request.json` and shows progress. So a dropped SSH session
doesn't stop a recording. LED4 on the board shows the state:

| LED4 | meaning |
|---|---|
| off | idle, waiting for a request |
| blue | countdown: get into position |
| red | recording |
| green | done, file saved |
| purple | bad request (see `app logs`) |

Files: `data/raw/ankle_<activity>_sNN_<YYYY-MM-DDTHH-MM>.csv` (board clock,
**UTC**), session number auto-incremented per activity, plus one row per
recording in `data/sessions.csv`. Unfinished files end in `.csv.partial`.

**Untethered.** Power the Q from the power bank and SSH in over Wi-Fi
(`ssh arduino@UNOQ3.local`). The App keeps running whatever happens to SSH.

**Run** (on the Q):

```bash
cd ~/ArduinoApps/activity_wearable
arduino-app-cli app start .                               # stops any other running app
python scripts/record.py sit --seconds 60 --countdown 10 --note "desk chair, feet flat"
python scripts/validate_recording.py --plot               # newest file -> PASS/WARN/FAIL + PNG
```

**Copy to the laptop and plot** (on the laptop, in the project folder):

```bash
python scripts/fetch_data.py --validate                   # scp from UNOQ3.local, validate + plot all
python scripts/validate_recording.py --show               # interactive plot of the newest file
```

What's validated: exact CSV columns, `seq`/MCU time strictly increasing (Linux
arrival time never going back), 100 Hz ±1 % with no dropped samples or gaps,
all channels changing, no samples within 2 LSB of ±16 g / ±2000 °/s (WARN if
any, FAIL above 1 %), mean |a| 0.8–1.2 g, little motion for sit/stand, and
duration vs requested.

Tests: `tests/test_stage3_recording.py` (synthetic) and
`tests/test_stage3_hardware.py` (skipped until a real `ankle_sit_*` file exists).

## Stage 4: collecting data (menu)

**Install:** nothing new in Python. Optional, on the Q: `sudo apt install tmux`
(see "If SSH drops" below).

The App is the default app now, so it starts by itself when the Q boots from
the power bank. Then, from the laptop:

```bash
ssh arduino@UNOQ3.local
cd ~/ArduinoApps/activity_wearable && source ~/.venvs/wearable/bin/activate
python scripts/collect.py                       # 120 s per recording, 10 s countdown
python scripts/collect.py --seconds 90          # other length
```

```
1) stand  2) walk  3) run  4) airbike  5) sit
Activity [ENTER = walk, q = quit]:
```

Type a number or a name (or just ENTER to repeat the last one). Then:

| Board shows | Phase | You |
|---|---|---|
| LED4 blue, matrix counts 10 → 1 | countdown | get into position |
| LED4 red, matrix = seconds left + progress bar | recording (beep in the terminal) | do the activity; no keyboard needed |
| LED4 green, matrix ✓ | done (beep) | come back; type a note or `x` to discard |
| matrix: one dim dot | idle | App running, waiting |

ENTER during the countdown cancels; ENTER during the recording stops early and
keeps what was recorded (`stopped_early = 1` in `sessions.csv`). After each
recording you get a quick validation, a note prompt, and a summary table of
sessions / minutes per activity.

**If SSH drops:** nothing is lost. The App does the recording, so it finishes
and logs the session (just without a note: add one later by editing
`data/sessions.csv`). Reconnect and run `collect.py` again. To keep the *menu*
itself alive across a dropped connection, run it inside tmux:
`tmux new -s collect` → `python scripts/collect.py`; after reconnecting,
`tmux attach -t collect`. (`nohup` doesn't fit here: the menu needs your keyboard.)

**Trimming:** the first and last `TRIM_SECONDS` (5 s) of every recording are
dropped when the data is loaded for processing (`wearable/dataset.py`), not in
the raw files. So a recording shorter than 12.6 s gives no windows at all;
`collect.py` warns you.

Tests: `tests/test_stage4_collection.py`: matrix frames, early stop,
sessions.csv edits (incl. migrating the Stage 3 file), summary, menu parsing,
trimming, and the App's request → countdown → record → stop/cancel → done
logic run against a fake Bridge.

## Stages 5-7: train, evaluate, and run both models

**Install:** nothing new. Training uses the test venv (scikit-learn, scipy,
matplotlib from `requirements-dev.txt`); the live App only needs numpy,
which the container already has. A C compiler (`sudo apt install gcc`) is only
needed if you want `tests/test_stage6_model2.py` to also compile-test the C export.

### Stage 5: evaluate on real data

```bash
python scripts/train.py                 # ~2-3 min on the Q; also fine on the laptop
python scripts/train.py --no-augment    # faster, skips the augmentation comparison
```

It uses **every** recording in `data/sessions.csv` (trimmed 5 s each end),
so re-run it after every batch of new recordings. It skips: files shorter than one window after trimming, and files that
FAIL validation. The classes are `config.CLASSES`: sit, stand, walk, run,
airbike (each recorded activity is its own class). Output in `reports/latest/`:

| File | What |
|---|---|
| `report.md` | everything below, plus "what to record next" |
| `confusion_model2_all_features.png` | RF on all ~70 features, session-grouped CV |
| per-recording table (in `report.md`) | each recording scored while held out, worst first: look at the worst one's plot first |
| `importances.png` | RF feature importances (chosen Model 2 features highlighted) |
| `learning_curve.png` | score vs. sessions per class: still rising = more data helps |

`reports/history.csv` gets one row per run, so you can watch the scores climb as you add data.

**Augmentation** (`wearable/augment.py`): ±10° random 3D rotation, ±10 %
motion amplitude, a little noise. Copies are added to *training* folds only
and keep their session id, so a test recording's copies never leak into
training. The final model uses augmentation only if it raised CV macro-F1.

### Stage 6: Model 2 (all IMU outputs), live

`train.py` picks k in {10, 15, 20, all} by grouped CV (ranking features
inside each fold, so no leakage), then saves:

- `models/model2.json`: the forest as plain numbers. **The App uses this.**
  It needs no scikit-learn, so laptop/Q version mismatches can't break it.
- `models/model2.joblib`: the scikit-learn object, for analysis on the
  laptop. This one **only loads with the same scikit-learn version**.

**Check versions** (run on both laptop and Q):

```bash
python -c "import sklearn; print(sklearn.__version__)"   # quick look
python scripts/check_models.py      # trained-with vs installed, and joblib == JSON predictions
pip install scikit-learn==<version printed>              # to match, if they differ
```

**Live** (on the Q; first restart the App once so it has the live-mode code:
`arduino-app-cli app restart ~/ArduinoApps/activity_wearable`):

```bash
python scripts/live.py               # Model 2; ENTER ends live mode
python scripts/live.py --model 1     # Model 1
python scripts/live.py --detach      # start and exit: keeps running untethered on the power bank
python scripts/live.py --stop
python scripts/live.py --replay data/raw/<file>.csv --model 2    # offline, laptop OK
```

Every 1.28 s the App classifies the last 2.56 s and shows the majority of
the last 3 predictions, so a change of activity shows up after ~3-5 s.
The LED matrix shows **SIT / STD / WLK / RUN / AIR**, a confidence bar on the
bottom row (share of the 3 votes), and 1 or 2 dots top-right for Model 1 or
2. LED4 is green. A new recording request ends live mode automatically.
Re-running `train.py` while live mode is on is fine: the App notices the new
model file within a second and switches to it (the log says "retrained: reloading").

**Live HUD web page** (probability graph + big activity banner): on the Q run

```bash
python scripts/dashboard.py          # prints http://UNOQ3.local:8000
```

and open that address on the laptop or a phone on the same Wi-Fi. The page has
ENGAGE MODEL 1 / 2 and DISENGAGE buttons (same as `live.py`), shows the shown
label (majority vote) as a banner, the newest window's probabilities as bars and
a confidence ring, and the last 90 s as a timeline (hover for values; the
"Data log" panel lists the numbers). It is a separate small server that only
reads `data/control/status.json`, so it never affects sampling; Ctrl-C stops it,
not live mode. To keep it running after SSH closes: `nohup python scripts/dashboard.py &`.
Files: `scripts/dashboard.py`, `web/dashboard.html`; tests: `tests/test_stage6_dashboard.py`.

**Optional, C export for the MCU:** `python scripts/export_c.py` writes
`models/model1.h` / `model2.h` (`int model2_predict(const float *x)`).
The features would also have to be computed in C on the MCU. That's a few
lines for Model 1 (one peak), and a lot of work for Model 2 (FFT, filters).
The script's docstring shows the m2cgen alternative.

### Stage 7: Model 1 (peak acceleration along ONE axis)

The feature is `acc_<axis>_peak` = max |accel| along one axis in the 2.56 s
window, gravity included. `train.py` tries x, y and z with tree depth 2 and 3
(grouped CV), keeps the best (preferring the shallower tree on a tie), and
prints the rules both as if/else and as plain ranges, e.g.

```
  acc_y_peak <= 1.40 g          -> stand
  1.40 < acc_y_peak <= 1.80 g   -> airbike
  1.80 < acc_y_peak <= 2.20 g   -> walk
  acc_y_peak >  2.20 g          -> run
```

`model1_axes.png` shows each axis's per-activity distribution with the
thresholds. The report's "What Model 1 can't separate" section lists the
pairs whose peak ranges overlap. Live: `python scripts/live.py --model 1`.

No range is narrower than `MODEL1_MIN_BAND_G` (0.05 g, in `config.py`): after
fitting, a narrower band is folded into its neighbour. Without this the tree
learned e.g. "0.97–0.99 g = stand", which only memorises the recordings (moving
the strap shifts a still axis by about that much).

Tests: `tests/test_stage5_evaluation.py`, `tests/test_stage6_model2.py`,
`tests/test_stage7_model1.py` (synthetic data; the `hardware`-marked ones use
your real recordings/models and skip until they exist).
