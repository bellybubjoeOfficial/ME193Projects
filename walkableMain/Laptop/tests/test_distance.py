"""Stage 8 -- distance: leg ruler, straight passes, swing angle, fits, live tracker."""

import numpy as np
import pandas as pd
import pytest

from laptop.config import load_config
from laptop.train.distance import (DistanceTracker, fit, lengths, px_per_m, sagittal_axis,
                                   step_features, straight_passes, stride_from_angle,
                                   swing_angle_deg)

CFG = load_config()
LEG_LM = CFG["distance"]["leg_length_m"] - CFG["distance"]["ankle_height_m"]


def landmarks(n=600, fps=30, ppm=200.0, speed_mps=1.2, turn_every_s=4.0):
    """Walker going back and forth; straight leg = LEG_LM * ppm px about a third of the time."""
    t = np.arange(n) / fps
    period = 2 * turn_every_s
    ph = (t % period) / period
    x = 100 + speed_mps * ppm * turn_every_s * np.where(ph < 0.5, 2 * ph, 2 - 2 * ph)
    rows = {"t_ns": (t * 1e9).astype(np.int64)}
    bend = 0.85 + 0.15 * np.abs(np.sin(np.pi * t * 1.9))       # leg length varies with knee bend
    for s in ("left", "right"):
        rows[f"{s}_hip_x"], rows[f"{s}_hip_y"] = x, np.full(n, 200.0)
        rows[f"{s}_ankle_x"], rows[f"{s}_ankle_y"] = x, 200 + LEG_LM * ppm * bend
        rows[f"{s}_hip_vis"] = rows[f"{s}_ankle_vis"] = np.full(n, 0.95)
    return pd.DataFrame(rows), x


def test_leg_ruler_recovers_scale():
    lm, _ = landmarks(ppm=180.0)
    assert px_per_m(lm, CFG) == pytest.approx(180.0, rel=0.02)


def test_straight_passes_measure_hip_travel():
    lm, x = landmarks(n=900, turn_every_s=4.0)
    passes = straight_passes(lm, CFG)
    assert len(passes) >= 5
    for a, b, travel in passes:
        assert travel == pytest.approx(abs(x[lm.t_ns == b][0] - x[lm.t_ns == a][0]), abs=15)
        assert (b - a) / 1e9 >= CFG["distance"]["min_straight_s"]


def test_swing_angle_of_a_known_rotation():
    rate = 100
    t = np.arange(110) / rate
    amp_deg = 35.0                                # shin swings +-35 deg -> 70 deg peak to peak
    omega = np.gradient(amp_deg * np.sin(2 * np.pi * t / 1.1), 1 / rate)
    gyro = np.column_stack([omega + 4.0, np.zeros_like(t), np.zeros_like(t)])  # + gyro bias
    theta = swing_angle_deg(gyro, rate, np.array([1.0, 0, 0]))
    assert theta == pytest.approx(70, abs=4)
    assert stride_from_angle(70, 1.0) == pytest.approx(2 * np.sin(np.radians(35)), rel=1e-6)


def test_sagittal_axis_finds_dominant_rotation():
    rng = np.random.default_rng(0)
    g = np.column_stack([rng.normal(0, 5, 1000), 200 * np.sin(np.arange(1000) / 20), rng.normal(0, 5, 1000)])
    ax = sagittal_axis(g)
    assert abs(ax[1]) > 0.99


def passes_from(L_true_fn, freqs, n_steps=5):
    out = []
    for f in freqs:
        feats = pd.DataFrame({"freq": np.full(n_steps, f), "pend": np.full(n_steps, 0.5 + 0.1 * f)})
        out.append((feats, n_steps * L_true_fn(f)))
    return out


def test_fits_recover_known_relations():
    passes = passes_from(lambda f: 0.2 + 0.25 * f, [1.4, 1.6, 1.8, 2.0, 2.2])
    c = fit("cadence", passes)
    assert c["a"] == pytest.approx(0.2, abs=1e-6) and c["b"] == pytest.approx(0.25, abs=1e-6)
    fixed = fit("fixed", passes)["L"]
    assert fixed == pytest.approx(np.mean([0.2 + 0.25 * f for f in [1.4, 1.6, 1.8, 2.0, 2.2]]))
    pend_passes = [(f, 1.3 * f["pend"].sum()) for f, _ in passes]
    assert fit("pendulum", pend_passes)["k"] == pytest.approx(1.3)
    assert (lengths("prior", {}, passes[0][0], CFG) == CFG["distance"]["prior_step_m"]).all()


def test_step_features_fill_and_pendulum():
    rate = 100
    t_imu = (np.arange(1200) * 1e7).astype(np.int64)
    gyro = np.zeros((1200, 3))
    steps_t = (np.arange(1, 11) * 0.55 * 1e9).astype(np.int64)
    legs = np.array(["R" if i % 2 == 0 else "L" for i in range(10)])
    for a in steps_t[legs == "R"]:   # each R stride: shin swings 60 deg
        i = int(a / 1e7)
        seg = np.arange(110)
        gyro[i:i + 110, 0] = np.gradient(30 * np.sin(2 * np.pi * seg / 110), 1 / rate)
    f = step_features(steps_t, legs, t_imu, gyro, rate, "R", np.array([1.0, 0, 0]), CFG)
    assert len(f) == 10 and f.freq.notna().all() and f.pend.notna().all()
    assert f.freq.iloc[5] == pytest.approx(1 / 0.55)
    expect = stride_from_angle(60, CFG["distance"]["leg_length_m"]) / 2
    assert f.pend.iloc[4] == pytest.approx(expect, rel=0.1)


def test_tracker_counts_and_band():
    tr = DistanceTracker("cadence", {"a": 0.1, "b": 0.27}, CFG, err_frac=0.05)
    for k in range(10):
        tr.add(int(k * 0.5e9), "LR"[k % 2])
    # first step uses the default frequency, the rest 2 Hz
    assert tr.total_m == pytest.approx(9 * (0.1 + 0.27 * 2) + (0.1 + 0.27 * 1.9))
    assert tr.band_m == pytest.approx(0.05 * tr.total_m)
    tr.reset()
    assert tr.total_m == 0 and tr.steps == 0


def test_walk_ratio_and_swing_speed_fits():
    freqs = [1.4, 1.7, 2.0, 2.3]
    passes = []
    for f in freqs:
        feats = pd.DataFrame({"freq": np.full(5, f), "pend": np.full(5, 0.5), "gpk": np.full(5, 100 * f)})
        passes.append((feats, 5 * 0.0055 * 60 * f))   # true walk ratio 0.0055
    assert fit("walk_ratio", passes)["r"] == pytest.approx(0.0055)
    sw = fit("swing_speed", [(f, 5 * (0.1 + 0.003 * f.gpk.iloc[0])) for f, _ in passes])
    assert sw["a"] == pytest.approx(0.1) and sw["b"] == pytest.approx(0.003)
    # the prior walk ratio is Apple's: 0.63 m at 114 steps/min
    one = pd.DataFrame({"freq": [114 / 60], "pend": [0.5], "gpk": [200.0]})
    assert lengths("walk_ratio_prior", {}, one, CFG)[0] == pytest.approx(0.63)
