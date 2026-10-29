"""Stage 2 models + session-grouped cross-validation.

WHY GROUPED: windows overlap by 50 %, and windows from the same recording share
the same person, strap position and speed. If a random split puts window 7 in
training and window 8 in testing, the model has effectively seen the test data,
and the score is far too optimistic. So every split keeps whole SESSIONS
(recordings) together: a session is either all-train or all-test.

We use StratifiedGroupKFold: it is GroupKFold (whole sessions per fold) that
also tries to give every fold a mix of classes, so no fold trains without
e.g. any running data.
"""

import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from wearable import config


def make_rf(**overrides):
    """The main model (Model 2): Random Forest with small-data defaults from config."""
    return RandomForestClassifier(**{**config.RF_PARAMS, **overrides})


def model_zoo(seed=config.RANDOM_SEED):
    """Models to compare. Scale-sensitive ones (SVM, kNN, logistic) get a StandardScaler.

    Trees (RF, boosting) don't care about feature scale, so they don't need one.
    """
    return {
        "random_forest": make_rf(random_state=seed),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.1, subsample=0.8, random_state=seed),
        "svm_rbf": make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", class_weight="balanced")),
        "knn": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "logistic": make_pipeline(StandardScaler(),
                                  LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced")),
    }


def group_splits(y, groups, n_splits=config.CV_FOLDS, seed=config.RANDOM_SEED):
    """Train/test index pairs that never split a session. Folds capped at #sessions."""
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        raise ValueError("need at least 2 sessions for grouped cross-validation")
    cv = StratifiedGroupKFold(n_splits=min(n_splits, n_groups), shuffle=True, random_state=seed)
    return list(cv.split(np.zeros(len(y)), y, groups))


def cross_validate(model, X, y, groups, n_splits=config.CV_FOLDS, seed=config.RANDOM_SEED):
    """Grouped cross-validation of one model.

    Returns a dict with out-of-fold predictions for every window (y_pred),
    per-fold accuracy and macro-F1, and fit time.
    """
    X, y, groups = np.asarray(X), np.asarray(y), np.asarray(groups)
    y_pred = np.empty_like(y)
    fold_acc, fold_f1 = [], []
    t0 = time.perf_counter()
    for train, test in group_splits(y, groups, n_splits, seed):
        assert not set(groups[train]) & set(groups[test]), "session leaked across train/test"
        m = clone(model).fit(X[train], y[train])
        y_pred[test] = m.predict(X[test])
        fold_acc.append(accuracy_score(y[test], y_pred[test]))
        fold_f1.append(f1_score(y[test], y_pred[test], average="macro"))
    return {
        "y_pred": y_pred,
        "fold_accuracy": np.array(fold_acc),
        "fold_macro_f1": np.array(fold_f1),
        "seconds": time.perf_counter() - t0,
    }


def compare_models(X, y, groups, models=None, n_splits=config.CV_FOLDS):
    """Run grouped CV for every model. Returns (summary table, {name: cv result})."""
    models = models or model_zoo()
    results, rows = {}, []
    for name, model in models.items():
        r = cross_validate(model, X, y, groups, n_splits)
        results[name] = r
        rows.append({
            "model": name,
            "accuracy": r["fold_accuracy"].mean(),
            "accuracy_std": r["fold_accuracy"].std(),
            "macro_f1": r["fold_macro_f1"].mean(),
            "seconds": r["seconds"],
        })
    table = pd.DataFrame(rows).sort_values("macro_f1", ascending=False, ignore_index=True)
    return table, results


def confusion_table(y_true, y_pred, labels=config.CLASSES):
    """Confusion matrix as a labelled table: rows = true class, columns = predicted."""
    labels = [c for c in labels if c in set(y_true) | set(y_pred)]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=[f"true_{c}" for c in labels], columns=[f"pred_{c}" for c in labels])
