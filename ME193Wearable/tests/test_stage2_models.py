"""Stage 2 model tests: grouped cross-validation + model comparison on synthetic data."""

import numpy as np
import pytest

from wearable import config, features, models, synth


@pytest.fixture(scope="module")
def dataset():
    """3 sessions x 5 activities x 30 s of synthetic data (~330 windows)."""
    ds = features.build_dataset(synth.make_sessions(sessions_per_activity=3, seconds=30))
    X = ds[features.feature_names()].to_numpy()
    return ds, X, ds["label"].to_numpy(), ds["session"].to_numpy()


# ---- Grouped splits (no leakage) --------------------------------------------

def test_splits_never_share_a_session(dataset):
    _, _, y, groups = dataset
    splits = models.group_splits(y, groups)
    assert len(splits) == config.CV_FOLDS
    tested = []
    for train, test in splits:
        assert not set(groups[train]) & set(groups[test])
        tested.extend(groups[test])
    # Every window is tested exactly once, and every session lands in a test fold.
    assert sorted(np.concatenate([t for _, t in splits])) == list(range(len(y)))
    assert set(tested) == set(groups)


def test_folds_capped_at_number_of_sessions():
    y = np.array(["walk", "walk", "run", "run"])
    groups = np.array(["a", "a", "b", "b"])
    assert len(models.group_splits(y, groups, n_splits=5)) == 2


def test_one_session_is_an_error():
    with pytest.raises(ValueError):
        models.group_splits(np.array(["walk"] * 4), np.array(["a"] * 4))


# ---- Models -------------------------------------------------------------------

def test_rf_uses_small_data_defaults():
    rf = models.make_rf()
    assert rf.max_depth == 8 and rf.min_samples_leaf == 3


def test_random_forest_separates_synthetic_classes(dataset):
    _, X, y, groups = dataset
    r = models.cross_validate(models.make_rf(), X, y, groups)
    assert r["fold_accuracy"].mean() >= 0.95
    assert set(r["y_pred"]) == set(config.CLASSES)


def test_comparison_harness_runs_all_models(dataset):
    _, X, y, groups = dataset
    table, results = models.compare_models(X, y, groups)
    assert set(table["model"]) == {"random_forest", "gradient_boosting", "svm_rbf", "knn", "logistic"}
    assert table["accuracy"].between(0, 1).all()
    assert list(table["macro_f1"]) == sorted(table["macro_f1"], reverse=True)
    for r in results.values():
        assert len(r["y_pred"]) == len(y)
    # Synthetic classes are easy: every model should do well.
    assert (table["accuracy"] >= 0.85).all()


def test_confusion_table(dataset):
    _, X, y, groups = dataset
    r = models.cross_validate(models.make_rf(), X, y, groups)
    cm = models.confusion_table(y, r["y_pred"])
    n = len(config.CLASSES)
    assert cm.shape == (n, n)
    assert cm.to_numpy().sum() == len(y)
    assert list(cm.index) == [f"true_{c}" for c in config.CLASSES]
