"""Stage 6 tests: feature selection, the JSON model format the App runs, the live
predictor (timing, voting, same features as training), and the C export.
"""

import shutil
import subprocess

import numpy as np
import pytest

from wearable import config, features, forest_lite, imu, live, models, synth, training


@pytest.fixture(scope="module")
def synth_ds():
    recs = synth.make_sessions(("stand", "walk", "run", "airbike"), sessions_per_activity=3, seconds=25)
    ds = features.build_dataset(recs)
    ds["copy"] = 0
    return ds, recs


@pytest.fixture(scope="module")
def rf_and_dict(synth_ds):
    ds, _ = synth_ds
    cols = features.feature_names()
    rf = models.make_rf(n_estimators=40).fit(ds[cols].to_numpy(), ds["label"].to_numpy())
    return rf, forest_lite.export(rf, cols, "model2", {"sklearn": "x"}), ds[cols].to_numpy()


# ---- JSON model = same model as scikit-learn ---------------------------------------

def test_json_forest_matches_sklearn_exactly(rf_and_dict):
    rf, d, X = rf_and_dict
    assert np.allclose(forest_lite.predict_proba(d, X), rf.predict_proba(X), atol=1e-5)
    assert (forest_lite.predict(d, X) == rf.predict(X)).all()


def test_json_round_trip(rf_and_dict, tmp_path):
    _, d, X = rf_and_dict
    forest_lite.save(d, tmp_path / "m.json")
    d2 = forest_lite.load(tmp_path / "m.json")
    assert d2["classes"] == d["classes"] and (forest_lite.predict(d2, X) == forest_lite.predict(d, X)).all()


def test_export_rejects_wrong_feature_names(rf_and_dict):
    rf, _, _ = rf_and_dict
    with pytest.raises(ValueError):
        forest_lite.export(rf, ["only_one"], "model2")


# ---- Feature selection ------------------------------------------------------------------

def test_select_features_picks_k_from_options(synth_ds):
    ds, _ = synth_ds
    cols = features.feature_names()
    chosen, table = training.select_features(ds, cols, k_options=[5, 10])
    assert len(chosen) in (5, 10) and set(chosen) <= set(cols)         # never "all": pruned
    assert list(table["n_features"]) == [5, 10, len(cols)]
    r = training.grouped_cv(models.make_rf(n_estimators=40), ds, chosen)
    assert r["fold_macro_f1"].mean() > 0.8


# ---- Live predictor -------------------------------------------------------------------

def test_numpy_gravity_filter_matches_scipy():
    acc = synth.make_recording("walk", 10, seed=5)[imu.ACCEL_COLS].to_numpy()
    a = features.gravity_lowpass(acc, use_scipy=True)
    b = features.gravity_lowpass(acc, use_scipy=False)
    assert np.max(np.abs(a - b)) < 1e-6


def test_live_timing_first_after_one_window_then_every_step(rf_and_dict):
    _, d, _ = rf_and_dict
    df = synth.make_recording("walk", 10, seed=6)
    p = live.LivePredictor(d)
    when = [i + 1 for i, row in enumerate(df[imu.CSV_COLUMNS].itertuples(index=False)) if p.add(row)]
    step = int(config.WINDOW_SAMPLES * (1 - config.WINDOW_OVERLAP))
    assert when[0] == config.WINDOW_SAMPLES
    assert np.all(np.diff(when) == step)                # every 1.28 s at 100 Hz


def test_live_majority_vote_smooths_one_odd_window(monkeypatch):
    """Scripted per-window predictions: a lone "run" is voted away; two in the last three win."""
    script = iter(["walk", "walk", "run", "walk", "run", "run"])
    classes = ["walk", "run"]
    monkeypatch.setattr(live.forest_lite, "predict_proba",
                        lambda model, X: np.array([[1.0, 0.0] if next(script) == "walk" else [0.0, 1.0]]))
    p = live.LivePredictor({"classes": classes, "features": ["acc_y_peak"], "trees": []}, vote=3)
    shown = [o for row in synth.make_recording("stand", 9.0, seed=0)[imu.CSV_COLUMNS].itertuples(index=False)
             if (o := p.add(row))]
    assert [o["raw_label"] for o in shown] == ["walk", "walk", "run", "walk", "run", "run"]
    assert [o["label"] for o in shown] == ["walk", "walk", "walk", "walk", "run", "run"]
    assert shown[2]["vote_fraction"] == pytest.approx(2 / 3)


def test_live_features_match_training_features(synth_ds):
    """Live filters gravity over a rolling 10 s buffer, training over the whole
    recording. Away from the recording start the window features must agree."""
    _, recs = synth_ds
    df = recs[3][0]                                       # a walk recording
    offline = features.recording_features(df)
    p = live.LivePredictor({"classes": [], "features": [], "trees": []})
    rows = df[imu.CSV_COLUMNS].to_numpy()
    n_end = config.WINDOW_SAMPLES + 9 * 128               # where an offline window ends, >10 s in
    for row in rows[:n_end]:
        p.buf.append([row[i] for i in live.ACC_IDX] + [row[i] for i in live.GYR_IDX])
    data = np.asarray(p.buf)
    grav = features.gravity_lowpass(data[:, :3], use_scipy=False)
    w = slice(len(data) - config.WINDOW_SAMPLES, len(data))
    live_f = features.window_features(data[w, :3], data[w, 3:], grav[w])
    start = (n_end - config.WINDOW_SAMPLES) / config.SAMPLE_RATE_HZ
    off = offline[np.isclose(offline["t_start_s"], start)].iloc[0]
    for name in ["acc_mag_std", "gyr_mag_dom_freq", "acc_y_peak", "body_mag_rms"]:
        assert live_f[name] == pytest.approx(off[name], rel=0.02, abs=0.01), name
    for name in ["grav_x", "grav_y", "grav_z"]:          # gravity edge effects: looser
        assert live_f[name] == pytest.approx(off[name], abs=0.05), name


def test_live_replay_recognises_synthetic_activities(rf_and_dict):
    _, d, _ = rf_and_dict
    for act in ["stand", "walk", "run", "airbike"]:
        preds = live.replay(d, synth.make_recording(act, 15, seed=500))   # a session it never saw
        labels = [x["label"] for x in preds]
        assert labels.count(config.ACTIVITY_TO_CLASS[act]) / len(labels) >= 0.8, (act, labels)
        assert set(preds[-1]) >= {"label", "raw_label", "vote_fraction", "proba", "n"}


# ---- C export ------------------------------------------------------------------------

def test_c_export_structure(rf_and_dict):
    _, d, _ = rf_and_dict
    src = forest_lite.to_c(d, "model2")
    assert src.count("static void model2_tree") == len(d["trees"])
    assert "int model2_predict(const float *x)" in src
    assert f"#define MODEL2_N_FEATURES {len(d['features'])}" in src


@pytest.mark.skipif(shutil.which("cc") is None and shutil.which("gcc") is None, reason="no C compiler")
def test_c_export_predicts_like_python(rf_and_dict, tmp_path):
    _, d, X = rf_and_dict
    small = {**d, "trees": d["trees"][:10]}
    (tmp_path / "model2.h").write_text(forest_lite.to_c(small, "model2"))
    (tmp_path / "main.c").write_text(
        '#include <stdio.h>\n#include "model2.h"\nint main(void){float x[MODEL2_N_FEATURES];'
        "while(1){for(int i=0;i<MODEL2_N_FEATURES;i++) if(scanf(\"%f\",&x[i])!=1) return 0;"
        'printf("%d\\n", model2_predict(x));}}\n')
    cc = shutil.which("cc") or shutil.which("gcc")
    subprocess.run([cc, "-O1", "-o", str(tmp_path / "m"), str(tmp_path / "main.c")], check=True)
    inp = "\n".join(" ".join(f"{v:.9g}" for v in row) for row in X[:200])
    out = subprocess.run([str(tmp_path / "m")], input=inp, capture_output=True, text=True, check=True).stdout
    want = [d["classes"].index(c) for c in forest_lite.predict(small, X[:200].astype(np.float32))]
    assert [int(v) for v in out.split()] == want


# ---- Real trained model (skipped until scripts/train.py has run) --------------------------

@pytest.mark.hardware
def test_real_model2_loads_and_replays():
    path = config.MODELS_DIR / "model2.json"
    if not path.exists():
        pytest.skip("run scripts/train.py first")
    d = forest_lite.load(path)
    assert d["kind"] == "model2" and set(d["classes"]) <= set(config.CLASSES)
    assert 1 <= len(d["features"]) <= 20 and set(d["features"]) <= set(features.feature_names())
    recs, _ = training.load_real_recordings()
    if not recs:
        pytest.skip("no recordings")
    preds = live.replay(d, recs[0][0])
    assert preds and all(p["label"] in d["classes"] for p in preds)
