"""IMU data handling: raw counts -> physical units, sketch info parsing, CSV I/O.

The sketch sends RAW int16 counts. Conversion lives here so it's tested.
"""

import numpy as np
import pandas as pd

from wearable import config

# Sensitivity (LSB per unit) from the MPU-6050 datasheet, keyed by full-scale range.
ACCEL_LSB_PER_G = {2: 16384.0, 4: 8192.0, 8: 4096.0, 16: 2048.0}
GYRO_LSB_PER_DPS = {250: 131.0, 500: 65.5, 1000: 32.8, 2000: 16.4}

# WHO_AM_I values. 0x68 is a genuine MPU-6050 (it reads 0x68 even when the
# I2C address is 0x69). The others are register-compatible for what we use
# (accel/gyro config + data registers), so they pass with a warning.
WHO_AM_I_EXPECTED = 0x68
WHO_AM_I_COMPATIBLE = {
    0x70: "MPU-6500",
    0x71: "MPU-9250",
    0x73: "MPU-9255",
}
IMU_ADDRESSES = (0x68, 0x69)            # AD0 pin low / high

# Expected config register read-back (bits 4:3 = full-scale select).
FS_BITS_16G = 0x18
FS_BITS_2000DPS = 0x18

ACCEL_COLS = ["ax_g", "ay_g", "az_g"]
GYRO_COLS = ["gx_dps", "gy_dps", "gz_dps"]
# One row per sample. t_mcu_us is unwrapped (monotonic); t_host_s is the
# Linux clock when the sample arrived (epoch seconds).
CSV_COLUMNS = ["seq", "t_mcu_us", "t_host_s"] + ACCEL_COLS + GYRO_COLS


def counts_to_g(raw, range_g=config.ACCEL_RANGE_G):
    return np.asarray(raw, dtype=float) / ACCEL_LSB_PER_G[range_g]


def counts_to_dps(raw, range_dps=config.GYRO_RANGE_DPS):
    return np.asarray(raw, dtype=float) / GYRO_LSB_PER_DPS[range_dps]


class Unwrapper:
    """Turns a wrapping uint32 counter (MCU micros(), seq) into a monotonic int.

    micros() wraps every 2**32 µs ≈ 71.6 min; a long logging session would see
    time jump backwards without this.
    """

    def __init__(self, bits=32):
        self.modulus = 1 << bits
        self.offset = 0
        self.last = None

    def __call__(self, value):
        value = int(value)
        if self.last is not None and value < self.last - self.modulus // 2:
            self.offset += self.modulus
        self.last = value
        return value + self.offset


def sample_row(seq, t_mcu_us, t_host_s, raw6):
    """Build one CSV row (list matching CSV_COLUMNS) from a Bridge sample."""
    a = counts_to_g(raw6[:3])
    g = counts_to_dps(raw6[3:])
    return [int(seq), int(t_mcu_us), round(float(t_host_s), 6)] + \
           [round(float(v), 5) for v in a] + [round(float(v), 4) for v in g]


def parse_info(text):
    """Parse the sketch's imu_info() string into a dict.

    Example input:
      ok=1;bus=Wire2;addr=0x68;who=0x68;accel_cfg=0x18;gyro_cfg=0x18;scan=Wire2:0x68|Wire1:
    Hex fields become ints (or None for "none"); scan becomes {bus: [addrs]}.
    """
    fields = {}
    for part in text.strip().split(";"):
        if "=" not in part:
            raise ValueError(f"bad imu_info field {part!r} in {text!r}")
        key, value = part.split("=", 1)
        fields[key] = value

    def hex_or_none(v):
        return None if v in ("none", "", None) else int(v, 16)

    scan = {}
    for entry in fields.get("scan", "").split("|"):
        if not entry:
            continue
        bus, _, addrs = entry.partition(":")
        scan[bus] = [int(a, 16) for a in addrs.split(",") if a]

    return {
        "ok": fields.get("ok") == "1",
        "bus": None if fields.get("bus") in (None, "none") else fields["bus"],
        "addr": hex_or_none(fields.get("addr")),
        "who_am_i": hex_or_none(fields.get("who")),
        "accel_cfg": hex_or_none(fields.get("accel_cfg")),
        "gyro_cfg": hex_or_none(fields.get("gyro_cfg")),
        "scan": scan,
        "raw": text.strip(),
    }


def load_csv(path):
    """Load a capture CSV and check it has exactly the expected columns."""
    df = pd.read_csv(path)
    if list(df.columns) != CSV_COLUMNS:
        raise ValueError(f"{path}: columns {list(df.columns)} != {CSV_COLUMNS}")
    return df
