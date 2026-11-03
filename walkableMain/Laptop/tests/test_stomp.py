"""Stage 4 -- stomp detection in IMU and video, residual / drift correction."""

import numpy as np
import pytest

from laptop.sync.stomp import G, correction, imu_stomp, video_stomp

MS = 1_000_000


def imu_trace(seconds=20, rate=100, stomps=(3.0,), stomp_g=8.0, walk_g=2.0, seed=0):
    """|a| = 1 g + noise, walking heel spikes (walk_g) from 6 s, stomps (stomp_g)."""
    rng = np.random.default_rng(seed)
    n = int(seconds * rate)
    t = np.arange(n) / rate
    dyn = rng.normal(0, 0.05, n)
    for ts in np.arange(6.0, seconds - 4, 0.55):           # walking
        dyn += walk_g * np.exp(-((t - ts) / 0.02) ** 2) * (t >= ts - 0.01)
    for ts in stomps:                                       # sharp onset at ts, decays
        dyn += stomp_g * np.clip((t - ts) / 0.01, 0, 1) * np.exp(-np.clip(t - ts, 0, None) / 0.03)
    acc = np.zeros((n, 3))
    acc[:, 2] = (1 + dyn) * G                               # all along z
    return (t * 1e9).astype(np.int64) + 10**12, acc


def test_imu_stomp_onset_time():
    t, acc = imu_trace(stomps=(3.0,))
    hit = imu_stomp(t, acc, t[0], t[0] + 5e9, min_g=3.0, onset_frac=0.5)
    assert hit is not None
    t_hit = (hit[0] - t[0]) / 1e9
    assert t_hit == pytest.approx(3.0, abs=0.012)  # within ~1 IMU sample of the true onset
    assert hit[1] > 5


def test_walking_spikes_are_not_stomps():
    t, acc = imu_trace(stomps=())
    assert imu_stomp(t, acc, t[0], t[-1], min_g=3.0) is None


def test_picks_the_stomp_not_a_walking_step():
    t, acc = imu_trace(seconds=30, stomps=(2.0, 26.0))
    start = imu_stomp(t, acc, t[0], t[0] + 8e9)
    end = imu_stomp(t, acc, t[-1] - 8e9, t[-1])
    assert (start[0] - t[0]) / 1e9 == pytest.approx(2.0, abs=0.012)
    assert (end[0] - t[0]) / 1e9 == pytest.approx(26.0, abs=0.012)


def heel_video(fps=30.0, seconds=4.0, impact_s=2.0, lift_px=60, ground=600.0, noise=1.5, seed=0):
    """Higher-heel y: on the ground, lifts ~0.3 s before impact, slams down at impact_s."""
    rng = np.random.default_rng(seed)
    n = int(seconds * fps)
    t = np.arange(n) / fps
    y = np.full(n, ground)
    up = (t > impact_s - 0.45) & (t < impact_s)
    ph = (t[up] - (impact_s - 0.45)) / 0.45
    y[up] = ground - lift_px * np.sin(np.pi * ph) ** 0.5    # quick drop at the end
    y += rng.normal(0, noise, n)
    impact_frame = int(np.ceil(impact_s * fps))
    return np.arange(n), (t * 1e9).astype(np.int64), y, impact_frame


def test_video_stomp_impact_frame():
    f, t, y, true_f = heel_video()
    hit = video_stomp(f, t, y, t_imu=int(2.0e9), leg_len_px=250, window_s=1.0)
    assert hit is not None
    assert abs(hit[0] - true_f) <= 1
    assert hit[2] == pytest.approx(60, abs=10)


def test_video_stomp_ignores_no_lift():
    f, t, y, _ = heel_video(lift_px=5)
    assert video_stomp(f, t, y, int(2.0e9), leg_len_px=250) is None


def test_video_stomp_tolerates_missing_frames():
    f, t, y, true_f = heel_video(seed=3)
    y[[10, 11, 70, 90]] = np.nan
    hit = video_stomp(f, t, y, int(2.0e9), leg_len_px=250)
    assert abs(hit[0] - true_f) <= 1


def test_correction_constant_and_linear():
    one = correction({"t_imu_ns": [10 * 10**9], "residual_ns": [40 * MS]})
    assert one(5 * 10**9) == 5 * 10**9 + 40 * MS
    two = correction({"t_imu_ns": [0, 100 * 10**9], "residual_ns": [40 * MS, 50 * MS]})
    assert two(50 * 10**9) == pytest.approx(50 * 10**9 + 45 * MS)
    assert two(100 * 10**9) == pytest.approx(100 * 10**9 + 50 * MS)


def test_video_stomp_slam_then_settle():
    # Real end stomp (2026-11-03): heel drops most of the way in one frame, then
    # MediaPipe's tracking eases it the last few px over 2 frames. Impact = the slam.
    f, t, y, true_f = heel_video(noise=0.5)
    y[true_f] -= 7
    y[true_f + 1] -= 3
    hit = video_stomp(f, t, y, int(2.0e9), leg_len_px=250)
    assert abs(hit[1] / 1e9 - true_f / 30) < 1 / 30  # within one frame of the slam


def test_implausible_residual_left_out_of_correction(tmp_path, monkeypatch):
    """A 600 ms 'residual' (video matched a walking step) must not warp the labels."""
    import json
    import laptop.sync.stomp as S
    hits = iter([(10_000_000_000, 9.0), (70_000_000_000, 11.0)])
    monkeypatch.setattr(S, "imu_stomp", lambda *a, **k: next(hits))
    monkeypatch.setattr(S, "heel_track", lambda *a, **k: (np.zeros(5), 250.0))
    vids = iter([(300, 10_020_000_000, 50.0), (2100, 69_380_000_000, 50.0)])  # +20 ms, -620 ms
    monkeypatch.setattr(S, "video_stomp", lambda *a, **k: next(vids))
    monkeypatch.setattr(S, "plot", lambda *a, **k: None)
    import pandas as pd
    monkeypatch.setattr(S, "load_imu", lambda f: pd.DataFrame({
        "t_laptop_ns": pd.array([0, 80_000_000_000], dtype="Int64"), "ax": [0, 0], "ay": [0, 0], "az": [9.8, 9.8]}))
    pd.DataFrame({"frame_idx": [0, 1], "t_ns": [0, 80_000_000_000]}).to_csv(tmp_path / "frames.csv", index=False)
    out = S.analyze(tmp_path)
    assert out["stomps"]["end"]["implausible"]
    assert json.load(open(tmp_path / "sync_correction.json"))["residual_ns"] == [20_000_000]
