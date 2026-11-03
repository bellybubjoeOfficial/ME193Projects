"""Sample decoding, unit conversion and session sequence numbers."""

import math
import struct

# One record of the MCU "imu_batch" blob (see sketch.ino):
#   uint32 seq | uint64 t_mcu_ns | int16 ax ay az gx gy gz   (little-endian, packed)
RECORD = struct.Struct("<IQ6h")

G = 9.80665
# MPU-6050 full-scale settings (AFS_SEL / FS_SEL bits 4:3 of ACCEL_CONFIG / GYRO_CONFIG).
ACCEL_RANGE_G = {0: 2, 1: 4, 2: 8, 3: 16}
GYRO_RANGE_DPS = {0: 250, 1: 500, 2: 1000, 3: 2000}
RAW_MAX = 32767


def decode_batch(blob):
    """bytes -> [(mcu_seq, t_mcu_ns, (ax, ay, az, gx, gy, gz) raw counts), ...]"""
    if len(blob) % RECORD.size:
        raise ValueError(f"imu_batch of {len(blob)} bytes is not a multiple of {RECORD.size}")
    return [(r[0], r[1], r[2:]) for r in RECORD.iter_unpack(blob)]


def ranges_from_cfg(accel_cfg, gyro_cfg):
    """Register read-backs -> (accel_range_g, gyro_range_dps)."""
    return ACCEL_RANGE_G[(accel_cfg >> 3) & 3], GYRO_RANGE_DPS[(gyro_cfg >> 3) & 3]


class Converter:
    """Raw counts -> m/s^2 and deg/s, for a given full-scale range."""

    def __init__(self, accel_range_g=16, gyro_range_dps=2000):
        self.accel_range_g = accel_range_g
        self.gyro_range_dps = gyro_range_dps
        self.a_scale = accel_range_g * G / 32768.0
        self.g_scale = gyro_range_dps / 32768.0

    def __call__(self, raw):
        a, s = self.a_scale, self.g_scale
        # 4 / 3 decimals: just below one LSB (0.0048 m/s^2, 0.061 deg/s at max range).
        return (round(raw[0] * a, 4), round(raw[1] * a, 4), round(raw[2] * a, 4),
                round(raw[3] * s, 3), round(raw[4] * s, 3), round(raw[5] * s, 3))


def saturated(raw):
    return any(v >= RAW_MAX or v <= -RAW_MAX for v in raw)


class SessionSeq:
    """MCU tick numbers -> session seq: starts at 0, never goes backwards.

    The MCU seq is the index of the timer tick a sample belongs to, so a gap
    there is a real lost sample and is kept as a gap here. If the MCU reboots
    mid-session (its seq jumps back), we continue from the last seq plus the
    number of periods that elapsed, so seq stays monotonic.
    """

    def __init__(self):
        self.last_mcu = None
        self.last_out = None
        self.last_t = None
        self.gaps = 0  # missing seq values so far (lost samples)

    def __call__(self, mcu_seq, t_ns, period_ns):
        if self.last_mcu is None:
            out = 0
        else:
            d = (mcu_seq - self.last_mcu) % 2**32
            if 0 < d < 2**31:
                out = self.last_out + d
            else:  # MCU restarted (or repeated a tick): estimate from time
                out = self.last_out + max(1, math.floor((t_ns - self.last_t) / period_ns + 0.5))
            self.gaps += out - self.last_out - 1
        self.last_mcu, self.last_out, self.last_t = mcu_seq, out, t_ns
        return out
