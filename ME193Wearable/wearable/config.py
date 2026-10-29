"""Project-wide settings. Change values here, not scattered through the code."""

from pathlib import Path

# Project root = the App folder (…/ArduinoApps/activity_wearable on the Q,
# /app inside the App Lab container, wherever you cloned it on the laptop).
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CHECK_DIR = DATA_DIR / "check"          # Stage 1 connection-check captures
RAW_DIR = DATA_DIR / "raw"              # Stage 3+: one CSV per recording
SESSIONS_CSV = DATA_DIR / "sessions.csv"   # one metadata row per recording
CONTROL_DIR = DATA_DIR / "control"      # request/status files between scripts and the App

# ---- Sensor / sampling ------------------------------------------------------
SAMPLE_RATE_HZ = 100
ACCEL_RANGE_G = 16                      # must match the sketch (AFS_SEL = 3)
GYRO_RANGE_DPS = 2000                   # must match the sketch (FS_SEL = 3)

# ---- Stage 1: connection check ---------------------------------------------
CHECK_SECONDS = 30                      # how long main.py records for the check

# Pass/fail thresholds for the "board sitting still" checks.
STILL_ACCEL_MAG_TOL_G = 0.10            # | |a| - 1 g | must be below this
STILL_ACCEL_STD_MAX_G = 0.02            # above this, the board probably moved
GYRO_BIAS_WARN_DPS = 5.0                # MPU-6050 zero-rate offset: a few °/s is normal…
GYRO_BIAS_FAIL_DPS = 20.0               # …datasheet tolerance is ±20 °/s
GYRO_NOISE_MAX_DPS = 1.0                # std per axis while still
RATE_TOL_FRAC = 0.01                    # mean rate within ±1 % of SAMPLE_RATE_HZ
MAX_GAP_PERIODS = 2.5                   # any dt above this = a missing sample
MAX_DUPLICATE_FRAC = 0.01               # identical consecutive rows => frozen sensor
MAX_BRIDGE_LAG_S = 0.5                  # host-vs-MCU clock drift over the capture

# ---- Stage 3: recordings ----------------------------------------------------
# Activities you can record (= the menu in collect.py). Biking is only
# recorded as air-biking (lying on your back).
RECORD_ACTIVITIES = ["stand", "walk", "run", "airbike", "sit"]
FILE_PREFIX = "ankle"                   # data/raw/ankle_sit_s01_2026-10-28T14-05.csv
DEFAULT_RECORD_SECONDS = 120
DEFAULT_COUNTDOWN_SECONDS = 10

# Recording validation. "Clipped" = within 2 LSB of full scale.
CLIP_MARGIN_ACCEL_G = 2 / 2048
CLIP_MARGIN_GYRO_DPS = 2 / 16.4
CLIP_FAIL_FRAC = 0.01                   # any clipped sample = WARN, above 1 % = FAIL
DURATION_TOL_S = 1.0                    # recorded length vs requested length
# Length of the MEAN accel vector over a whole recording ≈ 1 g whatever the
# activity (body motion averages out, gravity doesn't); far from 1 g means a
# unit or range bug. (Not the mean of |a|: that is ~1.4 g when running.)
REC_ACCEL_MAG_RANGE_G = (0.8, 1.2)
# "Still" activities (sit, stand): the leg should barely move.
STILL_ACTIVITIES = ("sit", "stand")
STILL_REC_ACCEL_STD_MAX_G = 0.05        # looser than the Stage 1 bench check: a person fidgets
STILL_REC_GYRO_RMS_MAX_DPS = 10.0

# ---- Stage 4: data collection -------------------------------------------------
# Cut from each end of every recording at PROCESSING time (the raw file keeps
# everything): getting into position at the start, reaching for the board at the end.
TRIM_SECONDS = 5

# ---- Stage 2: windows + features --------------------------------------------
WINDOW_SAMPLES = 256                    # 2.56 s at 100 Hz
WINDOW_OVERLAP = 0.5                    # 50 % -> a new window every 128 samples (1.28 s)
GRAVITY_CUTOFF_HZ = 0.3                 # low-pass that separates gravity from body motion
GRAVITY_FILTER_ORDER = 2                # Butterworth order (run forward+backward = zero phase)
SPECTRAL_BANDS_HZ = [(0, 1), (1, 3), (3, 10)]
DOM_FREQ_RANGE_HZ = (0.3, 15.0)         # search range for the dominant frequency
FFT_MIN_POINTS = 1024                   # zero-pad the FFT -> ~0.1 Hz frequency grid

# ---- Stage 2: activities + models -------------------------------------------
# The classes the models predict, and how each recorded activity maps onto them.
# Every recorded activity is its own class.
CLASSES = ["sit", "stand", "walk", "run", "airbike"]
ACTIVITY_TO_CLASS = {a: a for a in CLASSES}

# Random Forest defaults chosen for small datasets (a few hundred windows):
# shallow-ish trees + a minimum leaf size so it can't memorise single windows.
# n_jobs=1: one CPU core. Same results as using all cores (fixed random_state),
# and training on the Q can't starve the App of CPU (= dropped samples).
RF_PARAMS = dict(n_estimators=200, max_depth=8, min_samples_leaf=3,
                 class_weight="balanced", random_state=0, n_jobs=1)
CV_FOLDS = 5                            # capped at the number of sessions
RANDOM_SEED = 0

# ---- Stage 5: evaluation on real data ----------------------------------------
REPORTS_DIR = ROOT / "reports"          # reports/latest/report.md + plots after every training run
FEATURE_CACHE_DIR = ROOT / ".feature_cache"   # features per recording, so re-runs only process new files
MIN_SESSIONS_PER_CLASS = 2              # grouped CV needs >= 2 recordings of a class to test it at all
TARGET_SESSIONS_PER_CLASS = 6           # below this, the report asks for more recordings of that class
WEAK_RECALL = 0.85                      # below this, the class is flagged as "getting confused"

# Augmentation (training folds only; test folds always use the real recordings).
AUGMENT_COPIES = 2                      # extra randomized copies of every training recording
AUGMENT_ROTATION_DEG = 10               # random 3D rotation: strap mounted slightly differently
AUGMENT_SCALE = 0.10                    # ±10 % on body acceleration and gyro (gravity stays 1 g)
AUGMENT_NOISE_G = 0.01
AUGMENT_NOISE_DPS = 0.5

# ---- Stage 6: Model 2 (all IMU outputs) -----------------------------------------
MODELS_DIR = ROOT / "models"
FEATURE_COUNT_OPTIONS = [10, 15, 20]    # "keep the k most important features"; CV picks k

# Live inference (inside the App): predict every 128 samples (1.28 s) on the
# last 256, with gravity filtered over a longer rolling buffer, then a
# majority vote over the last few predictions.
LIVE_BUFFER_SECONDS = 10.0
LIVE_VOTE_WINDOWS = 3                   # 3 windows ≈ 3.8 s of history
LIVE_MODEL_FILES = {"model1": "model1.json", "model2": "model2.json"}

# ---- Stage 7: Model 1 (peak acceleration along ONE axis) ------------------------
MODEL1_DEPTHS = [2, 3]                  # shallow tree; CV picks the depth (ties -> the simpler one)
# No Model 1 band (range of peak values -> one class) narrower than this.
# Strap position shifts a still axis by a few hundredths of a g between
# sessions, so a narrower band is the tree memorising recordings, not physics.
MODEL1_MIN_BAND_G = 0.05
