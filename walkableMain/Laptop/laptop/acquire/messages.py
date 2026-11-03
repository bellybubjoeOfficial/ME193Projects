"""Parse the UNO Q's MQTT payloads (see README "Message contract").

Every parser raises ParseError on anything that doesn't match the contract,
so a bad message is counted and skipped instead of corrupting the data.
"""

import json
from dataclasses import dataclass

import numpy as np

FIELDS = ["ax", "ay", "az", "gx", "gy", "gz"]
ROW_LEN = 3 + len(FIELDS)  # seq, t_mono_ns, t_wall_ns, then the six channels


class ParseError(ValueError):
    pass


@dataclass
class ImuBatch:
    device: str
    session: str
    rate_hz: float
    seq: np.ndarray        # int64, shape (n,)
    t_mono_ns: np.ndarray  # int64, device monotonic clock
    t_wall_ns: np.ndarray  # int64, device wall clock
    data: np.ndarray       # float64, shape (n, 6): ax ay az [m/s^2], gx gy gz [deg/s]

    def __len__(self):
        return len(self.seq)

    def device_time(self, clock):
        """The sample timestamps on the clock the sync pongs use ('wall' or 'mono')."""
        return self.t_wall_ns if clock == "wall" else self.t_mono_ns


def _load(payload):
    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode()
    try:
        msg = json.loads(payload)
    except json.JSONDecodeError as e:
        raise ParseError(f"not JSON: {e}") from None
    if not isinstance(msg, dict):
        raise ParseError("payload is not a JSON object")
    return msg


def _require(msg, key, types):
    if key not in msg:
        raise ParseError(f"missing '{key}'")
    value = msg[key]
    bool_ok = types is bool
    if (isinstance(value, bool) and not bool_ok) or not isinstance(value, types):
        raise ParseError(f"'{key}' has wrong type {type(value).__name__}")
    return value


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def parse_imu_batch(payload):
    msg = _load(payload)
    device = _require(msg, "device", str)
    session = _require(msg, "session", str)
    rate_hz = _require(msg, "rate_hz", (int, float))
    if rate_hz <= 0:
        raise ParseError("rate_hz must be > 0")
    if _require(msg, "fields", list) != FIELDS:
        raise ParseError(f"fields must be {FIELDS}")
    samples = _require(msg, "samples", list)
    if not samples:
        raise ParseError("empty samples")

    for i, row in enumerate(samples):
        if not isinstance(row, list) or len(row) != ROW_LEN:
            raise ParseError(f"sample {i} must be a list of {ROW_LEN} numbers")
        if not all(_is_int(v) for v in row[:3]):
            raise ParseError(f"sample {i}: seq/t_mono_ns/t_wall_ns must be integers")
        if not all(_is_num(v) for v in row[3:]):
            raise ParseError(f"sample {i}: channel values must be numbers")

    # Timestamps are ~1e18 ns: keep them int64 (float64 would lose ~100 ns).
    ints = np.array([row[:3] for row in samples], dtype=np.int64)
    data = np.array([row[3:] for row in samples], dtype=np.float64)
    return ImuBatch(device, session, float(rate_hz),
                    ints[:, 0], ints[:, 1], ints[:, 2], data)


STATUS_KEYS = ["device", "online", "session", "rate_hz_measured", "samples_sent",
               "samples_buffered", "reconnects", "accel_range_g", "gyro_range_dps",
               "uptime_s"]


def parse_status(payload):
    """Status is display-only, so only device + online are required."""
    msg = _load(payload)
    _require(msg, "device", str)
    _require(msg, "online", bool)
    return {k: msg.get(k) for k in STATUS_KEYS}


@dataclass
class Pong:
    id: int
    t_laptop_ns: int     # t0: laptop send (echoed back)
    t_dev_recv_ns: int   # t1
    t_dev_send_ns: int   # t2


def parse_pong(payload):
    msg = _load(payload)
    vals = [_require(msg, key, int)
            for key in ("id", "t_laptop_ns", "t_dev_recv_ns", "t_dev_send_ns")]
    if vals[3] < vals[2]:
        raise ParseError("t_dev_send_ns before t_dev_recv_ns")
    return Pong(*vals)
