"""Stage 1 connection checks. Each check returns a CheckResult (PASS/WARN/FAIL).

`info` is the dict from imu.parse_info(); `df` is a capture loaded with
imu.load_csv() that was recorded with the board sitting still.
"""

from dataclasses import dataclass

import numpy as np

from wearable import config, imu

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str

    @property
    def ok(self):
        return self.status != FAIL


def check_device_found(info):
    name = "device found"
    found = {bus: [hex(a) for a in addrs] for bus, addrs in info["scan"].items() if addrs}
    if info["addr"] in imu.IMU_ADDRESSES:
        return CheckResult(name, PASS, f"{hex(info['addr'])} on {info['bus']} (scan: {found})")
    if found:
        return CheckResult(name, FAIL, f"devices found but none at 0x68/0x69: {found}")
    return CheckResult(name, FAIL, "no I2C devices on any bus")


def check_who_am_i(info):
    name = "WHO_AM_I"
    who = info["who_am_i"]
    if who == imu.WHO_AM_I_EXPECTED:
        return CheckResult(name, PASS, "0x68 = MPU-6050")
    if who in imu.WHO_AM_I_COMPATIBLE:
        return CheckResult(name, WARN, f"{hex(who)} = {imu.WHO_AM_I_COMPATIBLE[who]}, not an MPU-6050 "
                                       "(register-compatible for our purposes)")
    return CheckResult(name, FAIL, f"unexpected value {who if who is None else hex(who)}")


def check_ranges(info):
    name = "range config"
    a, g = info["accel_cfg"], info["gyro_cfg"]
    if a is not None and g is not None and \
            a & 0x18 == imu.FS_BITS_16G and g & 0x18 == imu.FS_BITS_2000DPS:
        return CheckResult(name, PASS, "accel ±16 g, gyro ±2000 °/s (read back from sensor)")
    return CheckResult(name, FAIL, f"read back accel_cfg={a}, gyro_cfg={g}; expected 0x18 / 0x18")


def check_values_update(df):
    name = "values update"
    cols = imu.ACCEL_COLS + imu.GYRO_COLS
    values = df[cols].to_numpy()
    if len(values) < 2:
        return CheckResult(name, FAIL, f"only {len(values)} samples")
    flat = [c for c in cols if np.std(df[c]) == 0]
    dup_frac = float(np.mean(np.all(np.diff(values, axis=0) == 0, axis=1)))
    if flat:
        return CheckResult(name, FAIL, f"channels never change: {flat}")
    if dup_frac > config.MAX_DUPLICATE_FRAC:
        return CheckResult(name, FAIL, f"{dup_frac:.1%} of samples identical to the previous one")
    return CheckResult(name, PASS, f"all 6 channels change; {dup_frac:.2%} repeated rows")


def check_accel_1g(df):
    name = "|accel| ≈ 1 g"
    mag = np.linalg.norm(df[imu.ACCEL_COLS].to_numpy(), axis=1)
    err = abs(mag.mean() - 1.0)
    detail = f"mean |a| = {mag.mean():.3f} g, std {mag.std():.4f} g"
    if err > config.STILL_ACCEL_MAG_TOL_G:
        return CheckResult(name, FAIL, detail + f" (off by {err:.3f} g)")
    if mag.std() > config.STILL_ACCEL_STD_MAX_G:
        return CheckResult(name, WARN, detail + " — board may have moved during capture")
    return CheckResult(name, PASS, detail)


def check_gyro_still(df):
    name = "gyro ≈ 0 when still"
    g = df[imu.GYRO_COLS].to_numpy()
    bias, noise = g.mean(axis=0), g.std(axis=0)
    detail = "bias [°/s] = " + ", ".join(f"{b:+.2f}" for b in bias) + \
             "; noise std = " + ", ".join(f"{s:.2f}" for s in noise)
    worst = np.abs(bias).max()
    if worst > config.GYRO_BIAS_FAIL_DPS or noise.max() > config.GYRO_NOISE_MAX_DPS:
        return CheckResult(name, FAIL, detail)
    if worst > config.GYRO_BIAS_WARN_DPS:
        return CheckResult(name, WARN, detail + " — bias within datasheet spec, we'll subtract it later")
    return CheckResult(name, PASS, detail)


def check_sample_rate(df):
    """Steady 100 Hz with nothing dropped.

    Uses the MCU timestamps (when the sample was taken), not the Linux arrival
    time (which has Bridge/USB jitter). Checks: no seq gaps, no long dt gaps,
    mean rate within ±1 %, and the Linux side keeps up (no growing backlog).
    """
    name = "sample rate 100 Hz"
    fs = config.SAMPLE_RATE_HZ
    period_us = 1e6 / fs
    seq = df["seq"].to_numpy()
    t = df["t_mcu_us"].to_numpy(dtype=float)
    if len(t) < 3:
        return CheckResult(name, FAIL, f"only {len(t)} samples")

    dseq = np.diff(seq)
    dt = np.diff(t)
    dropped = int(np.sum(dseq[dseq > 1] - 1))
    rate = (len(t) - 1) / ((t[-1] - t[0]) / 1e6)
    big_gaps = int(np.sum(dt > config.MAX_GAP_PERIODS * period_us))
    p99_jitter_ms = float(np.percentile(np.abs(dt - period_us), 99)) / 1000

    # Arrival delay (host clock - MCU clock). Constant offset is fine; if it
    # grows, Python is falling behind the MCU.
    lag = df["t_host_s"].to_numpy() - t / 1e6
    lag_growth = float(lag.max() - lag.min())

    detail = (f"{rate:.2f} Hz over {(t[-1] - t[0]) / 1e6:.1f} s, {len(t)} samples, "
              f"dropped {dropped}, gaps {big_gaps}, p99 jitter {p99_jitter_ms:.2f} ms, "
              f"bridge lag spread {lag_growth * 1000:.0f} ms")
    problems = []
    if np.any(dseq <= 0) or np.any(dt <= 0):
        problems.append("seq/time not strictly increasing")
    if dropped:
        problems.append(f"{dropped} dropped samples")
    if big_gaps:
        problems.append(f"{big_gaps} timing gaps")
    if abs(rate - fs) > config.RATE_TOL_FRAC * fs:
        problems.append(f"rate {rate:.2f} Hz not within ±{config.RATE_TOL_FRAC:.0%}")
    if lag_growth > config.MAX_BRIDGE_LAG_S:
        problems.append("Linux side falling behind the MCU")
    if problems:
        return CheckResult(name, FAIL, detail + " — " + "; ".join(problems))
    return CheckResult(name, PASS, detail)


def run_all(info, df):
    results = [check_device_found(info), check_who_am_i(info), check_ranges(info)]
    if df is None:
        results.append(CheckResult("capture", FAIL, "no capture file"))
        return results
    results += [check_values_update(df), check_accel_1g(df),
                check_gyro_still(df), check_sample_rate(df)]
    return results
