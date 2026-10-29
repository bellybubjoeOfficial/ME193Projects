"""Synthetic ankle-IMU recordings for testing the pipeline before real data exists.

Each activity is a simple caricature (NOT a realistic gait model):

  stand    gravity + sensor noise, nothing else
  walk     ~2.0 Hz: leg swing (gyro + smooth accel) plus a sharp heel impact
  run      ~2.8 Hz: bigger swing, much bigger impacts
  airbike  ~1.2 Hz: smooth pedalling rotation while lying on your back
           (gravity along the sensor's z axis), tiny impacts
  sit      like stand, but the shin is tilted

Sensor axes assumed: +y = up the shin (your marked arrow), z = out of the
sensor face, x = across the leg. The leg swings about x. An accelerometer at
rest reads +1 g pointing UP, so standing reads about (0, +1, 0) g.

Every session gets random cadence (±10 %), amplitude (±20 %) and mounting tilt
(a few degrees), so sessions differ the way real recordings do and grouped
cross-validation has something to do.
"""

import numpy as np
import pandas as pd

from wearable import config, imu

# cadence (Hz), swing amplitude (°), impact peak (g), base shin tilt from vertical (°)
PROFILES = {
    "stand":   dict(freq=0.0, swing_deg=0,  impact_g=0.0, tilt_deg=0),
    "sit":     dict(freq=0.0, swing_deg=0,  impact_g=0.0, tilt_deg=60),
    "walk":    dict(freq=2.0, swing_deg=20, impact_g=1.5, tilt_deg=0),
    "run":     dict(freq=2.8, swing_deg=35, impact_g=4.0, tilt_deg=10),
    "airbike": dict(freq=1.2, swing_deg=30, impact_g=0.05, tilt_deg=0),    # lying on back: see "up" below
}


def _rot_x(theta):
    """Rotation matrices about x for an array of angles (rad) -> (n, 3, 3)."""
    c, s = np.cos(theta), np.sin(theta)
    r = np.zeros((len(theta), 3, 3))
    r[:, 0, 0] = 1
    r[:, 1, 1], r[:, 1, 2] = c, -s
    r[:, 2, 1], r[:, 2, 2] = s, c
    return r


def make_recording(activity, seconds=60.0, fs=config.SAMPLE_RATE_HZ, seed=0):
    """One synthetic recording, in the same column format main.py writes."""
    rng = np.random.default_rng(seed)
    p = PROFILES[activity]
    n = int(seconds * fs)
    t = np.arange(n) / fs

    freq = p["freq"] * rng.uniform(0.9, 1.1)
    amp = rng.uniform(0.8, 1.2)
    phase = rng.uniform(0, 2 * np.pi)

    # Shin angle about x: fixed tilt + swing at the cadence (+ a slow wobble).
    swing = np.deg2rad(p["swing_deg"] * amp) * np.sin(2 * np.pi * freq * t + phase)
    wobble = np.deg2rad(2) * np.sin(2 * np.pi * 0.1 * t + rng.uniform(0, 2 * np.pi))
    theta = np.deg2rad(p["tilt_deg"] + rng.normal(0, 3)) + swing + wobble

    # Gravity seen by the sensor ("up" = +y in the world, rotated into the sensor frame).
    up = np.array([0.0, 1.0, 0.0])
    if activity == "airbike":                    # lying on your back: shin up, sensor faces up
        up = np.array([0.0, 0.0, 1.0])
    acc = np.einsum("nij,j->ni", _rot_x(-theta), up)

    if freq > 0:
        # Smooth swing acceleration along the leg's forward axis (z), at the cadence.
        acc[:, 2] += 0.3 * amp * p["swing_deg"] / 20 * np.sin(2 * np.pi * freq * t + phase)
        # Heel impacts: short Gaussian pulses (40 ms wide), once per cycle, mostly along +y.
        period = 1 / freq
        hits = np.arange(phase / (2 * np.pi) * period, seconds, period)
        pulse = np.zeros(n)
        for h in hits:
            pulse += np.exp(-0.5 * ((t - h) / 0.04) ** 2)
        # Balance each spike with a gentle deceleration over the rest of the
        # stride, so body acceleration averages to zero (the leg doesn't fly
        # off) and the mean accel vector stays at 1 g, as in real recordings.
        pulse -= pulse.mean()
        acc += np.outer(pulse * p["impact_g"] * amp, [0.2, 1.0, 0.3])

    gyr = np.zeros((n, 3))
    gyr[:, 0] = np.rad2deg(np.gradient(theta, 1 / fs))      # rotation about x = dθ/dt
    gyr += rng.normal(0, 0.5, gyr.shape) + rng.normal(0, 1.0, 3)   # noise + per-session bias
    acc += rng.normal(0, 0.01, acc.shape)

    # Clip like the real sensor (±16 g, ±2000 °/s).
    acc = np.clip(acc, -config.ACCEL_RANGE_G, config.ACCEL_RANGE_G)
    gyr = np.clip(gyr, -config.GYRO_RANGE_DPS, config.GYRO_RANGE_DPS)

    df = pd.DataFrame({
        "seq": np.arange(n),
        "t_mcu_us": (t * 1e6).astype(np.int64),
        "t_host_s": 1.7e9 + t,
    })
    for i, c in enumerate(imu.ACCEL_COLS):
        df[c] = acc[:, i]
    for i, c in enumerate(imu.GYRO_COLS):
        df[c] = gyr[:, i]
    return df[imu.CSV_COLUMNS]


def make_sessions(activities=("sit", "stand", "walk", "run", "airbike"), sessions_per_activity=4,
                  seconds=60.0, seed=config.RANDOM_SEED):
    """List of (df, activity, session_id) ready for features.build_dataset()."""
    out = []
    k = 0
    for activity in activities:
        for s in range(1, sessions_per_activity + 1):
            df = make_recording(activity, seconds, seed=seed * 1000 + k)
            out.append((df, activity, f"{activity}_s{s:02d}"))
            k += 1
    return out
