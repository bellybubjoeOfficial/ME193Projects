import random
import struct

import pytest

from streamer import config, imu
from streamer.clock import ClockMap
from streamer.ring import SampleRing


def test_decode_batch_roundtrip():
    rec = struct.Struct("<IQ6h")
    blob = rec.pack(7, 123456789012, 1, -2, 2048, 32767, -32768, 0) * 3
    out = imu.decode_batch(blob)
    assert len(out) == 3
    assert out[0] == (7, 123456789012, (1, -2, 2048, 32767, -32768, 0))


def test_decode_rejects_partial_record():
    with pytest.raises(ValueError):
        imu.decode_batch(b"\0" * 25)


def test_units_at_max_range():
    conv = imu.Converter(16, 2000)
    ax, ay, az, gx, gy, gz = conv((2048, -2048, 0, 16384, 0, -32768))
    assert ax == pytest.approx(9.80665, abs=1e-3)   # 2048 counts = 1 g at +-16 g
    assert ay == pytest.approx(-9.80665, abs=1e-3)
    assert gx == pytest.approx(1000.0, abs=1e-3)    # half scale of +-2000 deg/s
    assert gz == pytest.approx(-2000.0, abs=1e-3)


def test_ranges_from_register_readback():
    assert imu.ranges_from_cfg(0x18, 0x18) == (16, 2000)
    assert imu.ranges_from_cfg(0x00, 0x08) == (2, 500)


def test_saturation():
    assert imu.saturated((0, 0, 32767, 0, 0, 0))
    assert imu.saturated((0, 0, 0, -32768, 0, 0))
    assert not imu.saturated((32000, 0, 0, 0, 0, 0))


def test_session_seq_starts_at_zero_and_keeps_gaps():
    s = imu.SessionSeq()
    p = 10_000_000
    assert [s(m, m * p, p) for m in (500, 501, 502, 505, 506)] == [0, 1, 2, 5, 6]
    assert s.gaps == 2


def test_session_seq_survives_mcu_seq_wrap():
    s = imu.SessionSeq()
    p = 10_000_000
    assert [s(m, i * p, p) for i, m in enumerate((2**32 - 2, 2**32 - 1, 0, 1))] == [0, 1, 2, 3]
    assert s.gaps == 0


def test_session_seq_mcu_reboot_stays_monotonic():
    s = imu.SessionSeq()
    p = 10_000_000
    s(1000, 0, p)
    s(1001, p, p)
    # MCU reboots: its seq restarts at 0, about 2 s later
    out = s(0, p + 2_000_000_000, p)
    assert out == 1 + 200
    assert s(1, p + 2_010_000_000, p) == out + 1


def test_ring_full_batches_only_until_key_ends():
    r = SampleRing(100)
    a, b = ("s1", 100), ("s1", 50)
    r.extend([(a, i) for i in range(15)])
    assert r.take(10, a) == (a, list(range(10)))
    assert r.take(10, a) is None                    # 5 left, session still going: wait
    r.extend([(b, i) for i in range(3)])
    assert r.take(10, b) == (a, list(range(10, 15)))  # rate changed: old key flushes short
    assert r.take(10, None) == (b, [0, 1, 2])         # stopped: flush short


def test_ring_overflow_is_counted_not_silent(caplog):
    r = SampleRing(10)
    r.extend([("k", i) for i in range(15)])
    assert len(r) == 10 and r.dropped == 5
    assert r.take(10, "k")[1] == list(range(5, 15))
    assert "dropped 5" in caplog.text


def test_ring_push_front_restores_order():
    r = SampleRing(100)
    r.extend([("k", i) for i in range(20)])
    key, rows = r.take(10, "k")
    r.push_front(key, rows)
    assert r.take(10, "k")[1] == list(range(10))


def test_clock_map_recovers_offset_and_skew():
    rnd = random.Random(1)
    skew, boot = 50e-6, 123_456_789_000
    cm = ClockMap()
    for i in range(600):  # 10 min of 1 Hz syncs
        t0 = boot + i * 1_000_000_000 + rnd.randint(0, 1000)
        # asymmetric, sometimes very late replies (queued behind a batch)
        rtt = rnd.choice([1_500_000, 2_000_000, 25_000_000])
        t_read = t0 + rnd.randint(0, rtt)
        t_mcu = int((t_read - boot) * (1 + skew))
        assert cm.add(t0, t_mcu, t0 + rtt)
    assert cm.info()["skew_ppm"] == pytest.approx(-50, abs=5)  # MCU fast -> negative correction
    t_mcu = int((boot + 600e9 - boot) * (1 + skew))
    assert abs(cm.to_linux(t_mcu) - (boot + 600_000_000_000)) < 1_500_000  # < 1.5 ms


def test_clock_map_tracks_large_skew():
    """The UNO Q MCU clock was measured ~880 ppm slow: the fit must not clamp it."""
    cm = ClockMap()
    skew = -880e-6
    for i in range(0, 120):
        t_read = i * 1e9
        cm.add(int(t_read - 3e6), int(t_read * (1 + skew)), int(t_read + 3e6))
    assert cm.info()["skew_ppm"] == pytest.approx(880, abs=10)
    t_mcu = int(120e9 * (1 + skew))
    assert abs(cm.to_linux(t_mcu) - 120e9) < 1e6  # < 1 ms after 2 min


def test_clock_map_samples_stay_continuous_across_refits():
    """Refits move the fit by ms; mapped sample intervals must not step."""
    cm = ClockMap()
    skew, period = -880e-6, 10_000_000
    t_sample, out, worst = 0, [], 0
    for sec in range(120):
        t_read = sec * 1e9
        cm.add(int(t_read - 3e6), int(t_read * (1 + skew)), int(t_read + 3e6))
        for _ in range(100):
            out.append(cm.map_sample(int(t_sample * (1 + skew))))
            t_sample += period
    d = [b - a for a, b in zip(out, out[1:])]
    assert max(abs(x - period) for x in d) <= period * 0.006  # <= 60 us, no 20 ms steps
    assert abs(out[-1] - (t_sample - period)) < 1e6           # and converged to < 1 ms


def test_clock_map_detects_mcu_reboot():
    cm = ClockMap()
    assert cm.add(0, 5_000_000_000, 1_000_000)
    assert not cm.add(2_000_000_000, 1_000, 2_001_000_000)


def test_config_example_loads(tmp_path):
    cfg = config.load(config_path())
    assert cfg.rate_hz == 100 and cfg.batch_size == 10 and cfg.qos == 1
    assert cfg.buffer_seconds >= 60
    bad = tmp_path / "bad.toml"
    bad.write_text('[mqtt]\nhost = "x"\n[stream]\nrate_hz = 99\n')
    with pytest.raises(ValueError):
        config.load(bad)


def config_path():
    from pathlib import Path
    return Path(__file__).resolve().parent.parent / "config" / "streamer.toml"
