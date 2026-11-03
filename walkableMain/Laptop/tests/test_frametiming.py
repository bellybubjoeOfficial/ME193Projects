"""Stage 2 -- frame timing: fps, dropped-frame holes, stability windows."""

import numpy as np
import pytest

from laptop.vision.frametiming import (find_drops, find_late_stamps, fps_stability, mean_fps,
                                       window_fps)

P = 1e9 / 30  # 30 fps period in ns


def stamps(n, jitter_ms=0.0, seed=0, skip=()):
    rng = np.random.default_rng(seed)
    idx = np.array([i for i in range(n) if i not in set(skip)])
    return (idx * P + rng.normal(0, jitter_ms * 1e6, len(idx))).astype(np.int64) + 10**12


def test_clean_30fps():
    t = stamps(9000, jitter_ms=2)
    assert mean_fps(t) == pytest.approx(30, rel=1e-3)
    assert find_drops(t, 30) == []
    s = fps_stability(t, 10, 0.05)
    assert s["stable"] and len(s["window_fps"]) == 29


def test_single_and_multi_frame_holes():
    t = stamps(300, skip={100, 200, 201, 202})
    drops = find_drops(t, 30)
    assert [(d.after_frame, d.missing) for d in drops] == [(99, 1), (198, 3)]  # index among received frames
    assert drops[1].interval_ms == pytest.approx(4 * P / 1e6)


def test_jitter_is_not_a_drop():
    # 2 ms sd timestamp jitter (real webcams: ~1-3 ms) never reaches 1.5 periods
    t = np.sort(stamps(3000, jitter_ms=2, seed=3))
    assert find_drops(t, 30) == []


def test_camera_halving_fps_is_unstable():
    first = stamps(900)                                  # 30 s at 30 fps
    second = first[-1] + (np.arange(1, 451) * 2 * P).astype(np.int64)  # 30 s at 15 fps
    s = fps_stability(np.concatenate([first, second]), 10, 0.05)
    assert not s["stable"]
    assert s["window_fps"][0] == pytest.approx(30, abs=0.2)
    assert s["window_fps"][-1] == pytest.approx(15, abs=0.2)


def test_window_fps_drops_partial_window():
    assert len(window_fps(stamps(30 * 25), 10)) == 2
    assert fps_stability(stamps(30 * 5), 10, 0.05)["stable"] is False  # too short to judge


def test_late_stamp_is_not_a_drop():
    # Real webcam pattern: 57 ms then 8 ms -- frame on time, timestamp late.
    t = stamps(300)
    t[150] += int(24e6)
    assert find_drops(t, 30) == []
    late = find_late_stamps(t, 30)
    assert len(late) == 1 and late[0][0] == 150 and late[0][1] == pytest.approx(24, abs=0.1)


def test_late_stamp_next_to_real_hole_still_counts_the_hole():
    t = stamps(300, skip={200})
    assert [(d.after_frame, d.missing) for d in find_drops(t, 30)] == [(199, 1)]
    assert find_late_stamps(t, 30) == []
