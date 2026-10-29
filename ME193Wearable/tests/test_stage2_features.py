"""Stage 2 feature tests: windowing, gravity separation, features (synthetic data only)."""

import numpy as np
import pytest
from scipy import stats

from wearable import config, features, synth

FS = config.SAMPLE_RATE_HZ


def sine(freq, seconds=2.56, amp=1.0):
    t = np.arange(int(seconds * FS)) / FS
    return amp * np.sin(2 * np.pi * freq * t)


# ---- Windowing --------------------------------------------------------------

def test_window_config_is_2_56s_with_50pct_overlap():
    assert config.WINDOW_SAMPLES / FS == pytest.approx(2.56)
    assert config.WINDOW_OVERLAP == 0.5


@pytest.mark.parametrize("n,expected", [(0, 0), (255, 0), (256, 1), (383, 1), (384, 2), (1000, 6), (6000, 45)])
def test_window_counts(n, expected):
    starts = features.window_starts(n)
    assert len(starts) == expected
    if expected > 1:
        assert np.all(np.diff(starts) == 128)          # 50 % of 256
    if expected:
        assert starts[-1] + 256 <= n                   # never runs past the end


def test_recording_gives_one_row_per_window():
    df = synth.make_recording("walk", seconds=60)
    feats = features.recording_features(df)
    assert len(feats) == len(features.window_starts(len(df))) == 45
    assert feats["t_start_s"].iloc[1] == pytest.approx(1.28)


def test_too_short_recording_gives_empty_table():
    feats = features.recording_features(synth.make_recording("walk", seconds=2))
    assert len(feats) == 0
    assert list(feats.columns) == ["t_start_s"] + features.feature_names()


# ---- Feature table shape + no NaNs ------------------------------------------

def test_feature_names_unique_and_fixed():
    names = features.feature_names()
    assert len(names) == len(set(names))
    assert len(names) >= 60
    for required in ["acc_mag_mean", "acc_mag_std", "acc_mag_rms", "acc_mag_iqr", "acc_mag_skew",
                     "acc_mag_kurt", "body_mag_jerk_rms", "body_mag_zcr", "body_mag_dom_freq",
                     "body_mag_band_0_1hz", "body_mag_band_1_3hz", "body_mag_band_3_10hz",
                     "gyr_mag_spec_entropy", "gyr_mag_acf_peak", "body_sma", "body_corr_xy",
                     "gyr_corr_yz", "grav_x", "grav_y", "grav_z",
                     "acc_x_peak", "acc_y_peak", "acc_z_peak", "gyr_main_dom_freq"]:
        assert required in names


@pytest.mark.parametrize("activity", list(synth.PROFILES))
def test_every_activity_same_shape_no_nans(activity):
    feats = features.recording_features(synth.make_recording(activity, seconds=20))
    assert list(feats.columns) == ["t_start_s"] + features.feature_names()
    assert np.isfinite(feats.to_numpy()).all()


def test_flat_and_clipped_signals_give_no_nans():
    df = synth.make_recording("stand", seconds=10)
    flat = df.copy()
    flat[["ax_g", "ay_g", "az_g", "gx_dps", "gy_dps", "gz_dps"]] = 0.0     # dead sensor
    clipped = df.copy()
    clipped["ay_g"] = 16.0                                                 # pinned at full scale
    for d in (flat, clipped):
        assert np.isfinite(features.recording_features(d).to_numpy()).all()


# ---- Gravity separation -----------------------------------------------------

def test_gravity_lowpass_recovers_gravity_and_zero_mean_body():
    n = 3000
    t = np.arange(n) / FS
    g = np.array([0.2, 0.95, 0.1])
    motion = np.column_stack([0.5 * np.sin(2 * np.pi * 2 * t)] * 3)        # 2 Hz, well above 0.3 Hz
    grav = features.gravity_lowpass(g + motion)
    mid = slice(300, -300)                                                  # ignore filter edges
    assert np.allclose(grav[mid], g, atol=0.02)
    assert np.abs((g + motion - grav)[mid].mean(axis=0)).max() < 0.01


# ---- Single features on known signals ---------------------------------------

@pytest.mark.parametrize("freq", [1.2, 2.0, 2.8, 5.0])
def test_dominant_frequency_of_pure_sine(freq):
    assert features.dominant_frequency(sine(freq)) == pytest.approx(freq, abs=0.1)


def test_dominant_frequency_of_flat_signal_is_zero():
    assert features.dominant_frequency(np.ones(256)) == 0.0


@pytest.mark.parametrize("activity", ["walk", "run", "airbike"])
def test_cadence_recovered_from_synthetic_activity(activity):
    # The generator randomises cadence by ±10 % per session; allow for that.
    nominal = synth.PROFILES[activity]["freq"]
    feats = features.recording_features(synth.make_recording(activity, seconds=30, seed=3))
    assert feats["gyr_main_dom_freq"].median() == pytest.approx(nominal, rel=0.12)


def test_gyro_magnitude_peaks_at_double_cadence():
    # Documents WHY gyr_main_dom_freq exists: |sin| repeats twice per cycle.
    assert features.dominant_frequency(np.abs(sine(1.2) * 200)) == pytest.approx(2.4, abs=0.1)


def test_band_energies_split_variance():
    x = sine(2.0, amp=1.0) + sine(5.0, amp=0.5)
    e = features.band_energies(x)
    assert sum(e) == pytest.approx(np.var(x), rel=0.05)
    assert e[1] > 3 * e[2] > 0                       # 2 Hz (1–3 band) holds 4x the power of 5 Hz
    assert e[0] < 0.05 * sum(e)


def test_spectral_entropy_noise_high_sine_low():
    noise = np.random.default_rng(0).normal(size=256)
    assert features.spectral_entropy(noise) > 0.8
    assert features.spectral_entropy(sine(2.0)) < 0.5


def test_autocorr_peak_periodic_vs_noise():
    assert features.autocorr_peak(sine(2.0)) > 0.9
    assert features.autocorr_peak(np.random.default_rng(0).normal(size=256)) < 0.35


def test_zero_crossing_rate_of_sine_is_twice_frequency():
    assert features.zero_crossing_rate(sine(2.0, seconds=10)) == pytest.approx(4.0, abs=0.15)    # 39 or 40 crossings in 10 s


def test_skew_kurtosis_match_scipy():
    x = np.random.default_rng(1).gamma(2.0, size=256)
    assert features._safe_skew(x) == pytest.approx(stats.skew(x))
    assert features._safe_kurtosis(x) == pytest.approx(stats.kurtosis(x))


# ---- Features that should separate the activities ---------------------------

def _median(activity, col, seed=0):
    return features.recording_features(synth.make_recording(activity, 20, seed=seed))[col].median()


def test_posture_upright_vs_lying():
    assert _median("stand", "grav_y") > 0.9                 # gravity along the shin
    assert _median("airbike", "grav_z") > 0.9               # lying on back: gravity through the sensor face


def test_peak_acceleration_orders_stand_walk_run():
    assert _median("stand", "acc_y_peak") < _median("walk", "acc_y_peak") < _median("run", "acc_y_peak")


def test_dataset_label_is_the_recorded_activity():
    ds = features.build_dataset(synth.make_sessions(("sit", "airbike"), sessions_per_activity=1, seconds=10))
    assert set(ds["activity"]) == set(ds["label"]) == {"sit", "airbike"}
    assert ds["session"].nunique() == 2
