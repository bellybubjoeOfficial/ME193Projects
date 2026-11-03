"""Stage 6 -- dataset building, peak picking, scoring, and that the CNN learns."""

import numpy as np
import pytest

from laptop.config import load_config
from laptop.train.dataset import SessionData, imu_grid, labeled_mask, soft_targets
from laptop.train.events import pick_peaks, predict_events, score
from laptop.train.model import StepCNN

CFG = load_config()


def test_soft_targets_peak_at_event():
    t = (np.arange(1000) * 1e7).astype(np.int64)  # 100 Hz
    y = soft_targets(t, [int(2e9), int(5e9)], ["L", "R"], 0.03)
    assert y[200, 0] == pytest.approx(1.0) and y[500, 1] == pytest.approx(1.0)
    assert y[200, 1] == 0 and y[203, 0] == pytest.approx(np.exp(-0.5), rel=1e-3)
    assert y[230, 0] == 0  # beyond 4 sigma


def test_imu_grid_marks_lost_samples():
    import pandas as pd
    seq = np.array([10, 11, 12, 15, 16])
    df = pd.DataFrame({"seq": seq, "t_laptop_ns": pd.array(seq * 10_000_000, dtype="Int64"),
                       **{c: np.ones(5) for c in ["ax", "ay", "az", "gx", "gy", "gz"]}})
    t, x, valid = imu_grid(df)
    assert len(t) == 7 and valid.tolist() == [1, 1, 1, 0, 0, 1, 1]
    assert t[3] == 130_000_000 and x[3].sum() == 0


def test_labeled_mask_excludes_stomps_and_unseen():
    t = (np.arange(3000) * 1e7).astype(np.int64)        # 30 s
    frame_t = (np.arange(900) / 30 * 1e9).astype(np.int64)
    seen = np.ones(900, dtype=bool)
    seen[450:600] = False                                # heels hidden 15-20 s
    m = labeled_mask(t, frame_t, seen, [int(2e9), int(28e9)], margin_s=1.0)
    assert not m[:300].any() and not m[2701:].any()      # before start / after end stomp + margin
    assert m[1000] and not m[1700] and m[2200]


def test_pick_peaks_threshold_and_refractory():
    p = np.zeros(500)
    for i, h in [(100, 0.9), (110, 0.7), (200, 0.4), (300, 0.8)]:
        p[i - 2:i + 3] = [h * 0.5, h * 0.8, h, h * 0.8, h * 0.5]
    assert pick_peaks(p, 100, threshold=0.5, refractory_s=0.3) == [100, 300]


def test_score_matches_within_tolerance():
    truth = [(100, "L"), (150, "R"), (200, "L"), (250, "R")]
    pred = [(101, "L"), (152, "R"), (230, "L")]
    r = score(pred, truth, np.ones(400, bool), 100, tol_s=0.067)
    assert r["matched"] == 2 and r["truth"] == 4 and r["predicted"] == 3
    assert r["precision"] == pytest.approx(2 / 3) and r["recall"] == 0.5
    assert r["timing_median_abs_ms"] == pytest.approx(15)


def test_model_keeps_length():
    import torch
    out = StepCNN(ch=8)(torch.zeros(2, 6, 256))
    assert out.shape == (2, 2, 256)


def synthetic_session(name, seconds=60, seed=0):
    """Shin IMU: each strike = a sharp accel spike; L strikes on az, R (other leg) smaller on ay."""
    rng = np.random.default_rng(seed)
    n = seconds * 100
    t = (np.arange(n) * 1e7).astype(np.int64)
    x = rng.normal(0, 0.3, (n, 6)).astype(np.float32)
    x[:, 2] += 9.81
    base = np.arange(1.0, seconds - 1, 0.55)              # a strike every 0.55 s, alternating legs
    ts = base + rng.normal(0, 0.03, len(base))
    events = []
    for k, tt in enumerate(ts):
        i = int(tt * 100)
        leg = "L" if k % 2 == 0 else "R"
        if leg == "L":
            x[i:i + 4, 2] += [12, 6, 2, 0.5]
            x[i - 15:i, 3] += np.linspace(0, 80, 15)   # swing gyro before strike
        else:
            x[i:i + 4, 1] += [5, 3, 1, 0.3]
        events.append((i, leg))
    y = soft_targets(t, [t[i] for i, _ in events], [l for _, l in events], 0.03)
    return SessionData(name, t, x, np.ones(n, bool), np.ones(n, bool), y, events, {}, 100.0)


def test_cnn_learns_synthetic_strikes():
    from laptop.train.train import evaluate, train_model
    cfg = {**CFG, "train": {**CFG["train"], "channels": 16, "batch": 32}}
    train = [synthetic_session(f"s{i}", seed=i) for i in range(3)]
    held = synthetic_session("held", seed=9)
    model, norm = train_model(train, cfg, iterations=250, log_every=0)
    r, b, probs, pred = evaluate(model, norm, held, cfg)
    assert r["recall"] >= 0.9 and r["precision"] >= 0.9, r
    assert r["timing_median_abs_ms"] <= 20
