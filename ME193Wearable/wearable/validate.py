"""Stage 3 recording validation. Each check returns a checks.CheckResult (PASS/WARN/FAIL).

`df` is a recording loaded with imu.load_csv() (which already enforces the
column format). Sample rate / dropped samples / frozen-sensor checks are the
Stage 1 ones; this adds what matters for a real (moving) recording.
"""

import numpy as np

from wearable import checks, config, imu
from wearable.checks import FAIL, PASS, WARN, CheckResult


def check_monotonic(df):
    """seq and MCU time strictly increasing; Linux arrival time never going backwards."""
    name = "timestamps monotonic"
    if len(df) < 2:
        return CheckResult(name, FAIL, f"only {len(df)} samples")
    problems = []
    for col in ("seq", "t_mcu_us"):
        bad = int(np.sum(np.diff(df[col].to_numpy()) <= 0))
        if bad:
            problems.append(f"{col} goes backwards/repeats {bad}x")
    back = int(np.sum(np.diff(df["t_host_s"].to_numpy()) < 0))
    if back:
        problems.append(f"t_host_s goes backwards {back}x")
    if problems:
        return CheckResult(name, FAIL, "; ".join(problems))
    return CheckResult(name, PASS, "seq, t_mcu_us strictly increasing; t_host_s non-decreasing")


def clipped_fraction(df):
    """Fraction of samples at (or within 2 LSB of) full scale, per channel."""
    out = {}
    for cols, full, margin in [(imu.ACCEL_COLS, config.ACCEL_RANGE_G, config.CLIP_MARGIN_ACCEL_G),
                               (imu.GYRO_COLS, config.GYRO_RANGE_DPS, config.CLIP_MARGIN_GYRO_DPS)]:
        for c in cols:
            out[c] = float(np.mean(np.abs(df[c].to_numpy()) >= full - margin))
    return out


def check_clipping(df):
    name = "no clipping"
    frac = clipped_fraction(df)
    hit = {c: f for c, f in frac.items() if f > 0}
    if not hit:
        return CheckResult(name, PASS, f"nothing near ±{config.ACCEL_RANGE_G} g / ±{config.GYRO_RANGE_DPS} °/s")
    detail = ", ".join(f"{c} {f:.2%}" for c, f in hit.items())
    if max(hit.values()) > config.CLIP_FAIL_FRAC:
        return CheckResult(name, FAIL, f"clipped: {detail}")
    return CheckResult(name, WARN, f"a few clipped samples: {detail}")


def check_duration(df, expected_s):
    name = "duration"
    got = len(df) / config.SAMPLE_RATE_HZ
    detail = f"{got:.1f} s recorded, {expected_s:.0f} s requested"
    if abs(got - expected_s) > config.DURATION_TOL_S:
        return CheckResult(name, FAIL, detail)
    return CheckResult(name, PASS, detail)


def check_sensible_values(df, activity=None):
    """|mean accel vector| ≈ 1 g for any activity; for still activities, little motion too.

    It must be the length of the MEAN vector (= gravity: body motion averages
    out), not the mean of |a|: |a| is never negative, so impacts don't cancel
    and running at the ankle really does average ~1.4 g.
    """
    name = "sensible values"
    a = df[imu.ACCEL_COLS].to_numpy()
    g = df[imu.GYRO_COLS].to_numpy()
    mag = np.linalg.norm(a, axis=1)
    g_len = float(np.linalg.norm(a.mean(axis=0)))
    grav = a.mean(axis=0) / max(g_len, 1e-9)
    gyro_rms = float(np.sqrt(np.mean((g - g.mean(axis=0)) ** 2)))     # bias removed
    detail = (f"|mean a| {g_len:.3f} g; mean |a| {mag.mean():.3f} g, std {mag.std():.3f} g; "
              f"gyro rms {gyro_rms:.1f} °/s; "
              f"gravity dir (x,y,z) = ({grav[0]:+.2f}, {grav[1]:+.2f}, {grav[2]:+.2f})")
    lo, hi = config.REC_ACCEL_MAG_RANGE_G
    if not lo <= g_len <= hi:
        return CheckResult(name, FAIL, detail + f" — |mean a| outside {lo}–{hi} g")
    if activity in config.STILL_ACTIVITIES and (mag.std() > config.STILL_REC_ACCEL_STD_MAX_G or
                                                gyro_rms > config.STILL_REC_GYRO_RMS_MAX_DPS):
        return CheckResult(name, WARN, detail + f" — a lot of motion for '{activity}'")
    return CheckResult(name, PASS, detail)


def run_all(df, activity=None, expected_s=None):
    results = [check_monotonic(df), checks.check_sample_rate(df), checks.check_values_update(df),
               check_clipping(df), check_sensible_values(df, activity)]
    if expected_s is not None:
        results.append(check_duration(df, expected_s))
    return results
