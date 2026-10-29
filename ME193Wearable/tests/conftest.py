import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wearable import imu  # noqa: E402


def make_still_capture(seconds=30.0, fs=100.0, gravity=(0.0, 0.0, 1.0),
                       gyro_bias=(0.5, -0.8, 0.3), accel_noise=0.004, gyro_noise=0.15,
                       seed=0):
    """Synthetic capture of a board sitting still, in the same format main.py writes.

    Values are quantized to real sensor counts (±16 g, ±2000 °/s) so the
    "values update" and duplicate checks see realistic data.
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * fs)
    seq = np.arange(n) + 1000
    t_mcu = 5_000_000 + np.round(np.arange(n) * 1e6 / fs + rng.normal(0, 20, n)).astype(np.int64)
    t_host = 1.7e9 + t_mcu / 1e6 + 0.004 + rng.uniform(0, 0.003, n)

    a = np.array(gravity) + rng.normal(0, accel_noise, (n, 3))
    g = np.array(gyro_bias) + rng.normal(0, gyro_noise, (n, 3))
    a = np.round(a * imu.ACCEL_LSB_PER_G[16]) / imu.ACCEL_LSB_PER_G[16]
    g = np.round(g * imu.GYRO_LSB_PER_DPS[2000]) / imu.GYRO_LSB_PER_DPS[2000]

    df = pd.DataFrame({"seq": seq, "t_mcu_us": t_mcu, "t_host_s": t_host})
    for i, c in enumerate(imu.ACCEL_COLS):
        df[c] = a[:, i]
    for i, c in enumerate(imu.GYRO_COLS):
        df[c] = g[:, i]
    return df[imu.CSV_COLUMNS]


GOOD_INFO_TEXT = "ok=1;bus=Wire2;addr=0x68;who=0x68;accel_cfg=0x18;gyro_cfg=0x18;scan=Wire2:0x68|Wire1:"


@pytest.fixture
def still_df():
    return make_still_capture()


@pytest.fixture
def good_info():
    return imu.parse_info(GOOD_INFO_TEXT)
