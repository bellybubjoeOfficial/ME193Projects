"""Stage 7 tests: Model 1 = peak acceleration along ONE axis + a shallow tree.

Checks the feature itself (gravity included, sign-independent), axis/depth
selection by grouped CV, the human-readable rules, and that Model 1 runs
through the same live code as Model 2.
"""

import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from wearable import config, features, forest_lite, live, synth, training


# ---- The feature -----------------------------------------------------------------

def test_peak_includes_gravity_and_ignores_sign():
    n = config.WINDOW_SAMPLES
    acc = np.tile([0.0, -1.0, 0.0], (n, 1))              # still, y axis pointing DOWN
    acc[100, 1] = -3.0                                     # one heel strike along y
    z = np.zeros((n, 3))
    f = features.window_features(acc, z, acc)
    assert f["acc_y_peak"] == pytest.approx(3.0)
    assert f["acc_x_peak"] == 0 and f["acc_z_peak"] == 0


def test_model1_features_are_the_three_axis_peaks():
    assert training.MODEL1_FEATURES == {"x": "acc_x_peak", "y": "acc_y_peak", "z": "acc_z_peak"}
    assert set(training.MODEL1_FEATURES.values()) <= set(features.feature_names())


# ---- Axis + depth selection ---------------------------------------------------------

def _peaks_ds(seed=0, sessions=3, windows=8):
    """Fake feature table where ONLY the y peak separates the classes."""
    rng = np.random.default_rng(seed)
    centre = {"stand": 1.0, "walk": 2.0, "run": 4.0, "airbike": 1.5}
    rows = []
    for label, c in centre.items():
        for s in range(sessions):
            for _ in range(windows):
                rows.append({"session": f"{label}{s}", "label": label, "copy": 0,
                             "acc_x_peak": rng.uniform(0, 3), "acc_z_peak": rng.uniform(0, 3),
                             "acc_y_peak": c + rng.normal(0, 0.05)})
    return pd.DataFrame(rows)


def test_evaluate_model1_picks_the_separating_axis():
    table, axis, depth, cv = training.evaluate_model1(_peaks_ds())
    assert axis == "y" and depth in config.MODEL1_DEPTHS
    assert len(table) == 3 * len(config.MODEL1_DEPTHS)
    assert table.set_index(["axis", "depth"]).loc[("y", 3), "macro_f1"] > 0.95


def test_model1_tree_is_shallow():
    ds = _peaks_ds()
    for d in config.MODEL1_DEPTHS:
        m = training.fit_final(training.make_model1(d), ds, ["acc_y_peak"])
        assert m.get_depth() <= d <= 3


def test_separability_measures_overlap():
    sep = training.separability(_peaks_ds(), "acc_y_peak").set_index(["class_a", "class_b"])
    assert sep["overlap"].max() == 0                       # y separates every pair
    sep_x = training.separability(_peaks_ds(), "acc_x_peak")
    assert sep_x["overlap"].min() > 0.5                    # x is noise: everything overlaps


def _sliver_ds():
    """Class 'stand' only in a 0.02 g sliver inside 'sit': an unpruned tree carves it out."""
    rng = np.random.default_rng(1)
    rows = [{"session": f"sit{i % 3}", "label": "sit", "copy": 0, "acc_y_peak": v}
            for i, v in enumerate(np.r_[rng.uniform(0.85, 0.99, 30), rng.uniform(1.01, 1.3, 30)])]
    rows += [{"session": f"stand{i % 3}", "label": "stand", "copy": 0, "acc_y_peak": v}
             for i, v in enumerate(rng.uniform(0.995, 1.005, 30))]
    rows += [{"session": f"run{i % 3}", "label": "run", "copy": 0, "acc_y_peak": v}
             for i, v in enumerate(rng.uniform(2.5, 4, 30))]
    return pd.DataFrame(rows)


def test_model1_has_no_band_narrower_than_the_limit():
    ds = _sliver_ds()
    plain = DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, class_weight="balanced", random_state=0)
    plain = training.fit_final(plain, ds, ["acc_y_peak"])
    widths = lambda m: [hi - lo for lo, hi, _ in forest_lite.interval_rules(
        forest_lite.export(m, ["acc_y_peak"], "model1")) if np.isfinite(lo) and np.isfinite(hi)]
    assert min(widths(plain)) < config.MODEL1_MIN_BAND_G           # the problem exists unpruned
    m = training.fit_final(training.make_model1(3), ds, ["acc_y_peak"])
    assert all(w >= config.MODEL1_MIN_BAND_G for w in widths(m))
    d = forest_lite.export(m, ["acc_y_peak"], "model1")
    X = np.linspace(0.8, 4.5, 2000)[:, None]
    assert (forest_lite.predict(d, X) == m.predict(X)).all()         # sklearn, JSON agree after pruning
    assert "run" in forest_lite.predict(d, [[3.0]])                    # real splits survive


# ---- Human-readable rules -------------------------------------------------------------

@pytest.fixture
def model1_dict():
    ds = _peaks_ds()
    m = training.fit_final(training.make_model1(3), ds, ["acc_y_peak"])
    return forest_lite.export(m, ["acc_y_peak"], "model1")


def test_interval_rules_cover_the_line_and_match_predict(model1_dict):
    ranges = forest_lite.interval_rules(model1_dict)
    assert ranges[0][0] == -np.inf and ranges[-1][1] == np.inf
    assert all(a[1] == b[0] for a, b in zip(ranges[:-1], ranges[1:]))   # no gaps
    assert all(a[2] != b[2] for a, b in zip(ranges[:-1], ranges[1:]))   # neighbours merged
    assert [r[2] for r in ranges] == ["stand", "airbike", "walk", "run"]   # ordered by peak size
    for x in np.linspace(0, 6, 200):
        label = next(lab for lo, hi, lab in ranges if lo < x <= hi)
        assert forest_lite.predict(model1_dict, [[x]])[0] == label


def _edge_values(m1):
    """Values ON and a hair either side of every threshold: where rounding bugs show up."""
    thr = [t for t, left in zip(m1["trees"][0]["threshold"], m1["trees"][0]["left"]) if left != -1]
    return np.array([[t + e] for t in thr for e in (-1e-6, -1e-7, -1e-9, 0, 1e-9, 1e-7, 1e-6)])


def test_json_tree_branches_like_sklearn_at_the_thresholds():
    """Regression: sklearn rounds x to float32 before comparing; forest_lite must too."""
    ds = _peaks_ds()
    m = training.fit_final(training.make_model1(3), ds, ["acc_y_peak"])
    X = _edge_values(forest_lite.export(m, ["acc_y_peak"], "model1"))
    assert (forest_lite.predict(forest_lite.export(m, ["acc_y_peak"], "model1"), X) == m.predict(X)).all()


@pytest.mark.skipif(shutil.which("cc") is None and shutil.which("gcc") is None, reason="no C compiler")
def test_c_tree_branches_like_python_at_the_thresholds(model1_dict, tmp_path):
    (tmp_path / "model1.h").write_text(forest_lite.to_c(model1_dict, "model1"))
    (tmp_path / "main.c").write_text(
        '#include <stdio.h>\n#include "model1.h"\nint main(void){float x[1];'
        'while(scanf("%f",&x[0])==1) printf("%d\\n", model1_predict(x)); return 0;}\n')
    cc = shutil.which("cc") or shutil.which("gcc")
    subprocess.run([cc, "-O1", "-o", str(tmp_path / "m"), str(tmp_path / "main.c")], check=True)
    X = _edge_values(model1_dict).astype(np.float32)          # what the C code would be given
    inp = "\n".join(f"{v:.9g}" for v in X[:, 0])
    out = subprocess.run([str(tmp_path / "m")], input=inp, capture_output=True, text=True, check=True).stdout
    want = [model1_dict["classes"].index(c) for c in forest_lite.predict(model1_dict, X)]
    assert [int(v) for v in out.split()] == want


def test_rules_text_merges_splits_that_change_nothing():
    """sklearn may split a node into two leaves of the same class; print it as one rule."""
    tree = {"left": [1, -1, 3, -1, -1], "right": [2, -1, 4, -1, -1], "feature": [0, -2, 0, -2, -2],
            "threshold": [1.4, -2, 1.0, -2, -2],
            "value": [[0.5, 0.5], [0.0, 1.0], [0.9, 0.1], [1.0, 0.0], [0.6, 0.4]]}
    d = {"classes": ["stand", "walk"], "features": ["acc_y_peak"], "trees": [tree]}
    # node 2 splits into two 'stand' leaves -> one rule
    text = forest_lite.rules_text(d)
    assert text.count("-> stand") == 1 and text.count("if ") == 1
    assert "1.400 g" in text


def test_rules_text_rejects_forests(model1_dict):
    with pytest.raises(ValueError):
        forest_lite.rules_text({**model1_dict, "trees": model1_dict["trees"] * 2})


# ---- Live mode with Model 1 ---------------------------------------------------------------

def test_model1_runs_in_the_live_predictor(model1_dict):
    preds = live.replay(model1_dict, synth.make_recording("run", 8, seed=7))
    assert preds and all(p["label"] in model1_dict["classes"] for p in preds)


def test_c_export_of_model1_is_tiny(model1_dict):
    src = forest_lite.to_c(model1_dict, "model1")
    assert src.count("if (x[0] <=") <= 7 and "model1_predict" in src


# ---- Real trained model (skipped until scripts/train.py has run) -----------------------------

@pytest.mark.hardware
def test_real_model1_is_one_axis_and_shallow():
    path = config.MODELS_DIR / "model1.json"
    if not path.exists():
        pytest.skip("run scripts/train.py first")
    d = forest_lite.load(path)
    assert len(d["features"]) == 1 and d["features"][0] in training.MODEL1_FEATURES.values()
    assert len(d["trees"]) == 1 and d["meta"]["depth"] <= 3
    print("\n" + forest_lite.interval_text(d))
