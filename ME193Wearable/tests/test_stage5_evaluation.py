"""Stage 5 tests: loading real recordings, feature table + cache, grouped CV with
augmentation (no leakage), per-class / per-recording reports, augmentation physics.

Synthetic recordings written to a temp folder stand in for data/raw, so these
run anywhere. tests/test_stage5_evaluation.py::test_real_* use the real data
and are skipped until it exists.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from wearable import augment, config, features, imu, models, recording, synth, training


def _write(tmp_path, activity, session, df, seconds=None):
    raw = tmp_path / "raw"
    raw.mkdir(exist_ok=True)
    name = f"ankle_{activity}_s{session:02d}_2026-10-27T14-00.csv"
    df.to_csv(raw / name, index=False)
    recording.append_session(tmp_path / "sessions.csv", {
        "file": name, "activity": activity, "session": session, "seconds": seconds or len(df) / 100})
    return name


@pytest.fixture
def raw_folder(tmp_path):
    """2 sessions each of stand/walk/run/airbike (30 s) + one sit + one too-short run."""
    k = 0
    for act in ["stand", "walk", "run", "airbike"]:
        for s in (1, 2):
            _write(tmp_path, act, s, synth.make_recording(act, 30, seed=k))
            k += 1
    _write(tmp_path, "sit", 1, synth.make_recording("sit", 30, seed=99))
    _write(tmp_path, "run", 3, synth.make_recording("run", 11, seed=98))
    return tmp_path


def _load(folder):
    return training.load_real_recordings(folder / "raw", folder / "sessions.csv")


# ---- Loading -------------------------------------------------------------------

def test_load_keeps_every_class_skips_too_short(raw_folder):
    recs, notes = _load(raw_folder)
    assert len(recs) == 9
    assert {a for _, a, _ in recs} == {"sit", "stand", "walk", "run", "airbike"}
    assert any("run_s03" in n and "shorter than one window" in n for n in notes)


def test_load_skips_activities_that_are_not_classes(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ACTIVITY_TO_CLASS", {"walk": "walk"})
    _write(tmp_path, "sit", 1, synth.make_recording("sit", 30, seed=5))
    recs, notes = _load(tmp_path)
    assert recs == [] and "isn't a model class" in notes[0]


def test_load_trims_both_ends(raw_folder):
    recs, _ = _load(raw_folder)
    n_trim = config.TRIM_SECONDS * config.SAMPLE_RATE_HZ
    assert all(len(df) == 30 * config.SAMPLE_RATE_HZ - 2 * n_trim for df, _, _ in recs)


def test_running_is_not_rejected_for_high_mean_accel(tmp_path):
    """Regression: mean |a| is ~1.4 g when running (impacts never cancel), but the
    length of the MEAN accel vector is still ~1 g. Only the latter may be checked."""
    df = synth.make_recording("run", 30, seed=1)
    mag = np.linalg.norm(df[imu.ACCEL_COLS].to_numpy(), axis=1)
    assert mag.mean() > 1.2                     # the case that used to be wrongly skipped
    _write(tmp_path, "run", 1, df)
    recs, notes = _load(tmp_path)
    assert len(recs) == 1, notes


def test_load_skips_failed_validation(tmp_path):
    df = synth.make_recording("walk", 30, seed=2)
    df[imu.ACCEL_COLS] *= 9.81                  # m/s² instead of g: a unit bug
    _write(tmp_path, "walk", 1, df)
    recs, notes = _load(tmp_path)
    assert recs == [] and "FAILED validation" in notes[0]


def test_missing_file_is_noted(raw_folder):
    (raw_folder / "raw" / "ankle_walk_s01_2026-10-27T14-00.csv").unlink()
    recs, notes = _load(raw_folder)
    assert len(recs) == 8 and any("missing" in n for n in notes)


# ---- Feature table + cache --------------------------------------------------------

def test_feature_table_real_and_copies(raw_folder):
    recs, _ = _load(raw_folder)
    ds = training.build_feature_table(recs[:2], copies=2, cache_dir=None)
    assert set(ds["copy"]) == {0, 1, 2}
    per_copy = ds.groupby(["session", "copy"]).size()
    assert per_copy.nunique() == 1              # copies have as many windows as the original
    assert not ds[features.feature_names()].isna().any().any()
    assert set(ds["label"]) <= set(config.CLASSES)


def test_feature_cache_reused(raw_folder, tmp_path):
    recs, _ = _load(raw_folder)
    cache = tmp_path / "cache"
    a = training.build_feature_table(recs[:2], copies=1, cache_dir=cache)
    files = sorted(cache.glob("*.pkl"))
    assert len(files) == 4
    mtimes = [f.stat().st_mtime_ns for f in files]
    b = training.build_feature_table(recs[:2], copies=1, cache_dir=cache)
    assert [f.stat().st_mtime_ns for f in sorted(cache.glob("*.pkl"))] == mtimes   # read, not rewritten
    pd.testing.assert_frame_equal(a, b)


def test_check_enough_data_drops_thin_classes():
    ds = pd.DataFrame({"session": ["a", "b", "c", "d", "e"], "label": ["walk", "walk", "run", "run", "airbike"],
                       "copy": 0})
    kept, counts, dropped = training.check_enough_data(ds)
    assert dropped == ["sit", "stand", "airbike"] and set(kept["label"]) == {"walk", "run"}
    with pytest.raises(training.NotEnoughData):
        training.check_enough_data(ds[ds["label"] != "run"])


# ---- Grouped CV with augmented copies: no leakage -----------------------------------

class SessionSpy(BaseEstimator, ClassifierMixin):
    """Fake model: column 0 is the session number. Fails if it is asked to predict
    a session it was trained on (that would be leakage)."""

    def fit(self, X, y):
        self.seen_ = set(X[:, 0])
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        assert not set(X[:, 0]) & self.seen_, "a test session (or its copy) was in the training set"
        return np.full(len(X), self.classes_[0])


def test_grouped_cv_never_trains_on_test_sessions_or_their_copies():
    rows = []
    for s in range(8):
        for copy in range(3):
            for w in range(4):
                rows.append({"session": f"s{s}", "sid": s, "label": ["walk", "run"][s % 2], "copy": copy})
    ds = pd.DataFrame(rows)
    r = training.grouped_cv(SessionSpy(), ds, ["sid"], use_copies=True)
    assert len(r["y_pred"]) == 8 * 4            # scored on the real windows only


# ---- Reports ----------------------------------------------------------------------

def test_reports_on_known_predictions():
    y = np.array(["walk"] * 4 + ["run"] * 4)
    p = np.array(["walk"] * 4 + ["run", "run", "walk", "walk"])
    rep = training.per_class_report(y, p)
    assert list(rep.index) == ["walk", "run"]
    assert rep.loc["run", "recall"] == 0.5 and rep.loc["walk", "precision"] == pytest.approx(4 / 6)
    assert training.confused_pairs(y, p) == [("run", "walk", 0.5)]
    ds = pd.DataFrame({"session": ["w1"] * 4 + ["r1", "r1", "r2", "r2"], "label": y, "copy": 0})
    t = training.per_session_table(ds, p)
    assert t.iloc[0]["session"] == "r2" and t.iloc[0]["accuracy"] == 0
    assert t.iloc[0]["predicted"] == "walk 2"


def test_advice_names_missing_and_confused_classes():
    counts = pd.Series({"sit": 6, "stand": 4, "walk": 3, "run": 1, "airbike": 6})
    rep = pd.DataFrame({"recall": [1.0, 0.5, 1.0]}, index=["stand", "walk", "airbike"])
    tips = training.advice(counts, ["run"], rep, [("walk", "airbike", 0.5)],
                           pd.DataFrame({"mean": [0.6, 0.8]}))
    text = "\n".join(tips)
    assert "Record 'run'" in text and "'walk' recall is only 50%" in text and "still rising" in text


# ---- Augmentation -------------------------------------------------------------------

def test_rotation_is_a_proper_rotation_within_limit():
    rng = np.random.default_rng(0)
    for _ in range(20):
        r = augment.random_rotation(rng, 10)
        assert np.allclose(r @ r.T, np.eye(3)) and np.isclose(np.linalg.det(r), 1)
        angle = np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1)))
        assert angle <= 10 + 1e-9


def test_augmented_copy_keeps_gravity_and_shape():
    df = synth.make_recording("walk", 30, seed=3)
    out = augment.augment_recording(df, np.random.default_rng(1))
    assert list(out.columns) == list(df.columns) and len(out) == len(df)
    for d in (df, out):
        g = np.linalg.norm(d[imu.ACCEL_COLS].to_numpy().mean(axis=0))
        assert g == pytest.approx(1.0, abs=0.05)                       # still 1 g of gravity
    assert not np.allclose(out[imu.ACCEL_COLS], df[imu.ACCEL_COLS])    # but actually changed
    pd.testing.assert_series_equal(out["t_mcu_us"], df["t_mcu_us"])


def test_augmented_copies_are_reproducible_and_keep_session_id():
    recs = [(synth.make_recording("run", 20, seed=4), "run", "run_s01")]
    a, b = augment.augmented_copies(recs, copies=2), augment.augmented_copies(recs, copies=2)
    assert [s for _, _, s in a] == ["run_s01", "run_s01"]
    pd.testing.assert_frame_equal(a[0][0], b[0][0])
    assert not a[0][0].equals(a[1][0])


# ---- Full Stage 5 harness on synthetic data ---------------------------------------------

def test_compare_models_on_synthetic_feature_table(raw_folder):
    recs, _ = _load(raw_folder)
    ds = training.build_feature_table(recs, copies=0, cache_dir=None)
    ds, _, _ = training.check_enough_data(ds)
    real = ds[ds["copy"] == 0]
    cols = features.feature_names()
    table, results = models.compare_models(real[cols].to_numpy(), real["label"].to_numpy(),
                                           real["session"].to_numpy())
    assert set(table["model"]) == {"random_forest", "gradient_boosting", "svm_rbf", "knn", "logistic"}
    assert table.loc[table["model"] == "random_forest", "macro_f1"].iloc[0] > 0.8


# ---- Real data (skipped until recordings exist) -----------------------------------------

@pytest.mark.hardware
def test_real_recordings_load_and_cover_classes():
    recs, notes = training.load_real_recordings()
    if not recs:
        pytest.skip("no usable real recordings yet")
    labels = {config.ACTIVITY_TO_CLASS[a] for _, a, _ in recs}
    assert len(labels) >= 2, notes
    for df, _, s in recs:
        assert len(df) >= config.WINDOW_SAMPLES, s
