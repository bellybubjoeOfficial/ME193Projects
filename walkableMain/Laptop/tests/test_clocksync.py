"""Stage 1 -- NTP-style offset / round-trip math and the running median."""

import random

import numpy as np
import pytest

from laptop.sync.clocksync import ClockSync, ntp_offset_rtt

MS = 1_000_000


def exchange(offset, out_ms, back_ms, hold_ms=0.5, t0=10_000 * MS):
    """Simulate one ping/pong: device clock = laptop clock + offset."""
    t1 = t0 + int(out_ms * MS) + offset
    t2 = t1 + int(hold_ms * MS)
    t3 = t2 - offset + int(back_ms * MS)
    return t0, t1, t2, t3


def test_symmetric_delay_is_exact():
    off, rtt = ntp_offset_rtt(*exchange(offset=5_000 * MS, out_ms=4, back_ms=4))
    assert off == 5_000 * MS
    assert rtt == 8 * MS  # device hold time is excluded


def test_negative_offset():
    off, _ = ntp_offset_rtt(*exchange(offset=-123_456_789, out_ms=3, back_ms=3))
    assert off == -123_456_789


def test_asymmetric_delay_error_is_half_the_asymmetry():
    off, rtt = ntp_offset_rtt(*exchange(offset=0, out_ms=10, back_ms=2))
    assert rtt == 12 * MS
    assert off == 4 * MS  # (10 - 2) / 2


def test_realistic_device_timestamps_no_overflow():
    big = 1_730_000_000 * 10**9  # device wall-ish clock vs laptop monotonic
    off, rtt = ntp_offset_rtt(*exchange(offset=big, out_ms=5, back_ms=5))
    assert off == big and rtt == 10 * MS


def test_median_rejects_spikes():
    random.seed(1)
    cs = ClockSync(median_window=31, max_rtt_ms=250)
    true_off = 42 * MS
    for i in range(31):
        if i % 5 == 0:  # every 5th ping hits a slow, lopsided path
            cs.add(*exchange(true_off, out_ms=80, back_ms=2, t0=i * 2000 * MS))
        else:
            d = random.uniform(2, 6)
            cs.add(*exchange(true_off, out_ms=d, back_ms=d + random.uniform(-0.5, 0.5),
                             t0=i * 2000 * MS))
    assert abs(cs.offset_ns - true_off) < 0.5 * MS
    assert 3 * MS < cs.rtt_ns < 13 * MS


def test_huge_rtt_is_logged_but_not_used():
    cs = ClockSync(max_rtt_ms=100)
    s = cs.add(*exchange(7 * MS, out_ms=200, back_ms=200))
    assert not s.accepted and cs.offset_ns is None and not cs.synced
    cs.add(*exchange(7 * MS, out_ms=3, back_ms=3))
    assert cs.offset_ns == 7 * MS and len(cs.history) == 2


def test_device_to_laptop_on_arrays():
    cs = ClockSync()
    cs.add(*exchange(1_000 * MS, out_ms=2, back_ms=2))
    dev = np.array([1_500 * MS, 1_510 * MS], dtype=np.int64)
    np.testing.assert_array_equal(cs.device_to_laptop(dev), [500 * MS, 510 * MS])


def test_drift_estimate():
    cs = ClockSync()
    ppm = 20
    for i in range(90):  # 3 minutes of pings
        t0 = i * 2000 * MS
        cs.add(*exchange(int(t0 * ppm * 1e-6), out_ms=3, back_ms=3, t0=t0))
    assert cs.drift_ppm() == pytest.approx(ppm, abs=0.5)


def test_drift_needs_enough_span():
    cs = ClockSync()
    for i in range(5):
        cs.add(*exchange(0, out_ms=3, back_ms=3, t0=i * 2000 * MS))
    assert cs.drift_ppm() is None  # only 8 s of data
    assert cs.drift_ppm(min_span_s=5) is not None
