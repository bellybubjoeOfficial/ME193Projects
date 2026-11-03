"""Stage 1 -- parsing the UNO Q payloads (no broker needed)."""

import json

import numpy as np
import pytest

from laptop.acquire.messages import ParseError, parse_imu_batch, parse_pong, parse_status

T0 = 1_730_000_000_123_456_789  # realistic ns timestamp (> 2**53)


def batch(**overrides):
    msg = {
        "device": "unoq-shin", "session": "abc-123", "rate_hz": 100,
        "fields": ["ax", "ay", "az", "gx", "gy", "gz"],
        "units": {"accel": "m/s^2", "gyro": "deg/s"},
        "samples": [[s, T0 + s * 10_000_000, T0 + 7 + s * 10_000_000,
                     0.1, 9.8, -0.2, 1.5, -2, 3] for s in range(5)],
    }
    msg.update(overrides)
    return json.dumps(msg)


def test_valid_batch():
    b = parse_imu_batch(batch())
    assert len(b) == 5
    assert b.device == "unoq-shin" and b.session == "abc-123" and b.rate_hz == 100
    np.testing.assert_array_equal(b.seq, np.arange(5))
    assert b.data.shape == (5, 6)
    assert b.data[0, 1] == pytest.approx(9.8)


def test_timestamps_keep_full_ns_precision():
    b = parse_imu_batch(batch())
    assert b.t_mono_ns.dtype == np.int64
    assert int(b.t_mono_ns[0]) == T0  # would be off by ~100 ns as float64
    assert int(b.t_wall_ns[3]) == T0 + 7 + 30_000_000


def test_accepts_bytes():
    assert len(parse_imu_batch(batch().encode())) == 5


@pytest.mark.parametrize("bad", [
    "not json",
    "[1, 2, 3]",
    batch(samples=[]),
    batch(samples=[[0, 1, 2, 0, 0, 0, 0, 0]]),            # 8 values, need 9
    batch(samples=[[0, 1, 2, 0, 0, 0, 0, 0, 0, 0]]),      # 10 values
    batch(samples=[[0.5, 1, 2, 0, 0, 0, 0, 0, 0]]),       # float seq
    batch(samples=[[0, 1, 2, "x", 0, 0, 0, 0, 0]]),       # string channel
    batch(samples=[[True, 1, 2, 0, 0, 0, 0, 0, 0]]),      # bool is not an int
    batch(samples=[[0, 1, 2, None, 0, 0, 0, 0, 0]]),
    batch(fields=["ax", "ay", "az"]),
    batch(fields=["ay", "ax", "az", "gx", "gy", "gz"]),   # wrong order
    batch(rate_hz=0),
    batch(rate_hz="100"),
])
def test_rejects_malformed_batches(bad):
    with pytest.raises(ParseError):
        parse_imu_batch(bad)


@pytest.mark.parametrize("key", ["device", "session", "rate_hz", "fields", "samples"])
def test_rejects_missing_keys(key):
    msg = json.loads(batch())
    del msg[key]
    with pytest.raises(ParseError):
        parse_imu_batch(json.dumps(msg))


def test_pong():
    p = parse_pong(json.dumps({"id": 7, "t_laptop_ns": 1, "t_dev_recv_ns": 5, "t_dev_send_ns": 6}))
    assert (p.id, p.t_laptop_ns, p.t_dev_recv_ns, p.t_dev_send_ns) == (7, 1, 5, 6)


@pytest.mark.parametrize("bad", [
    {"id": 7, "t_laptop_ns": 1, "t_dev_recv_ns": 5},                          # missing
    {"id": 7, "t_laptop_ns": 1.5, "t_dev_recv_ns": 5, "t_dev_send_ns": 6},     # float
    {"id": 7, "t_laptop_ns": 1, "t_dev_recv_ns": 9, "t_dev_send_ns": 6},       # send < recv
])
def test_rejects_bad_pong(bad):
    with pytest.raises(ParseError):
        parse_pong(json.dumps(bad))


def test_status_partial_ok_and_missing_online_rejected():
    s = parse_status(json.dumps({"device": "unoq-shin", "online": True, "uptime_s": 12}))
    assert s["online"] is True and s["uptime_s"] == 12 and s["reconnects"] is None
    with pytest.raises(ParseError):
        parse_status(json.dumps({"device": "unoq-shin"}))
