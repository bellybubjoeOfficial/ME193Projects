"""Stages 5-7: evaluate on real data, train Model 2 (all IMU outputs) and Model 1 (one axis).

Everything here is re-run from scratch by scripts/train.py, on EVERY recording
listed in data/sessions.csv. So the system improves as you add data:
record more sessions, run train.py again, and the report, the chosen
features, the chosen axis/depth for Model 1, and both saved models all update.
reports/history.csv keeps one row per run, so you can watch the scores
change as the dataset grows, and the learning curve shows whether more
data is still helping.

Feature extraction is cached per recording (.feature_cache/), so a re-run
only processes new or changed files.

Cross-validation is always grouped by session (see models.py for why).
"""

import hashlib
import inspect
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import classification_report, f1_score
from sklearn.tree import DecisionTreeClassifier

from wearable import augment, config, dataset, features, imu, models, recording, validate
from wearable.checks import FAIL, WARN


class NotEnoughData(Exception):
    """Raised when there isn't enough real data to train/evaluate anything yet."""


META_COLS = ["session", "activity", "label", "copy", "t_start_s"]


# ---- Loading real recordings ----------------------------------------------------

def load_real_recordings(raw_dir=config.RAW_DIR, sessions_csv=config.SESSIONS_CSV,
                         trim_s=config.TRIM_SECONDS):
    """Every usable recording: list of (df, activity, session) + a list of skip/warn notes.

    Skipped: activities that aren't a model class (e.g. sit), files missing
    from data/raw, recordings too short after trimming, and recordings that
    FAIL validation (gaps, clipping, impossible values). WARNs are kept.
    """
    recs, notes = [], []
    for row in recording.read_sessions(sessions_csv):
        name, activity = row["file"], row["activity"]
        if activity not in config.ACTIVITY_TO_CLASS:
            notes.append(f"{name}: '{activity}' is recorded but isn't a model class -> not used")
            continue
        path = Path(raw_dir) / name
        if not path.exists():
            notes.append(f"{name}: listed in sessions.csv but the file is missing -> skipped")
            continue
        df = dataset.trim(imu.load_csv(path), trim_s)
        if len(df) < config.WINDOW_SAMPLES:
            notes.append(f"{name}: shorter than one window after trimming {trim_s} s each end -> skipped")
            continue
        results = validate.run_all(df, activity)
        bad = [r for r in results if r.status == FAIL]
        if bad:
            notes.append(f"{name}: FAILED validation ({'; '.join(r.name + ': ' + r.detail for r in bad)}) -> skipped")
            continue
        for r in results:
            if r.status == WARN:
                notes.append(f"{name}: warning, {r.name}: {r.detail} (kept)")
        recs.append((df, activity, path.stem))
    return recs, notes


# ---- Feature table (with cache) ---------------------------------------------------

def _feature_version():
    """Changes whenever the feature code or a setting that affects features changes."""
    src = inspect.getsource(features) + inspect.getsource(augment)
    settings = (config.WINDOW_SAMPLES, config.WINDOW_OVERLAP, config.GRAVITY_CUTOFF_HZ,
                config.SPECTRAL_BANDS_HZ, config.TRIM_SECONDS, config.AUGMENT_ROTATION_DEG,
                config.AUGMENT_SCALE, config.AUGMENT_NOISE_G, config.AUGMENT_NOISE_DPS)
    return hashlib.sha1((src + repr(settings)).encode()).hexdigest()[:12]


def _recording_table(df, activity, session, copy):
    if copy > 0:
        df = augment.augment_recording(df, augment.copy_rng(session, copy - 1))
    feats = features.recording_features(df)
    feats.insert(0, "session", session)
    feats.insert(1, "activity", activity)
    feats.insert(2, "label", config.ACTIVITY_TO_CLASS[activity])
    feats.insert(3, "copy", copy)                       # 0 = the real recording, 1.. = augmented copies
    return feats


def build_feature_table(recs, copies=0, cache_dir=config.FEATURE_CACHE_DIR):
    """One row per window, for the real recordings (copy 0) plus `copies` augmented copies each.

    With cache_dir, each (recording, copy) is stored as a pickle named after
    the recording's content hash, so a re-run skips everything already done.
    """
    version = _feature_version()
    cache_dir = Path(cache_dir) if cache_dir else None
    parts = []
    for df, activity, session in recs:
        content = hashlib.sha1(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()[:12]
        for copy in range(copies + 1):
            path = cache_dir / f"{session}__{copy}__{content}__{version}.pkl" if cache_dir else None
            if path is not None and path.exists():
                parts.append(pd.read_pickle(path))
                continue
            table = _recording_table(df, activity, session, copy)
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                table.to_pickle(path)
            parts.append(table)
    if not parts:
        return pd.DataFrame(columns=META_COLS + features.feature_names())
    return pd.concat(parts, ignore_index=True)


def check_enough_data(ds, min_sessions=config.MIN_SESSIONS_PER_CLASS):
    """Keep classes with >= min_sessions real recordings. Returns (ds, counts, dropped classes).

    Raises NotEnoughData if fewer than 2 classes are left (nothing to tell apart).
    """
    real = ds[ds["copy"] == 0]
    counts = real.groupby("label")["session"].nunique().reindex(config.CLASSES, fill_value=0)
    keep = [c for c in config.CLASSES if counts[c] >= min_sessions]
    dropped = [c for c in config.CLASSES if c not in keep]
    if len(keep) < 2:
        have = ", ".join(f"{c}: {counts[c]}" for c in config.CLASSES)
        raise NotEnoughData(f"need >= {min_sessions} recordings of at least 2 classes "
                            f"({'/'.join(config.CLASSES)}); have {have}")
    return ds[ds["label"].isin(keep)].reset_index(drop=True), counts, dropped


# ---- Grouped cross-validation (optionally with augmented training copies) --------------

def grouped_cv(model, ds, cols, use_copies=False, n_splits=config.CV_FOLDS):
    """Session-grouped CV on the REAL windows. With use_copies, each training fold also
    gets the augmented copies of its own training sessions (never of the test sessions).

    Returns {"y_true", "y_pred", "fold_macro_f1", "fold_accuracy"} for the real windows.
    """
    real = ds[ds["copy"] == 0].reset_index(drop=True)
    copies = ds[ds["copy"] > 0]
    X, y, g = real[cols].to_numpy(), real["label"].to_numpy(), real["session"].to_numpy()
    y_pred = np.empty_like(y)
    f1s, accs = [], []
    for train, test in models.group_splits(y, g, n_splits):
        Xtr, ytr = X[train], y[train]
        if use_copies and len(copies):
            extra = copies[copies["session"].isin(set(g[train]))]
            Xtr = np.vstack([Xtr, extra[cols].to_numpy()])
            ytr = np.concatenate([ytr, extra["label"].to_numpy()])
        m = clone(model).fit(Xtr, ytr)
        y_pred[test] = m.predict(X[test])
        f1s.append(f1_score(y[test], y_pred[test], average="macro"))
        accs.append(np.mean(y[test] == y_pred[test]))
    return {"y_true": y, "y_pred": y_pred, "fold_macro_f1": np.array(f1s), "fold_accuracy": np.array(accs)}


def per_class_report(y_true, y_pred):
    """precision / recall / f1 / windows per class, as a table."""
    labels = [c for c in config.CLASSES if c in set(y_true)]
    rep = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    return pd.DataFrame({c: rep[c] for c in labels}).T.rename(columns={"support": "windows"})


def per_session_table(ds, y_pred):
    """Out-of-fold accuracy for each recording, worst first.

    A recording that is mostly predicted as another class is either a hard
    case (a slow jog that looks like walking) or a mislabelled / bad file.
    Either way it's the first one to look at.
    """
    real = ds[ds["copy"] == 0].reset_index(drop=True)
    rows = []
    for s, d in real.assign(pred=y_pred).groupby("session"):
        counts = d["pred"].value_counts()
        rows.append({"session": s, "label": d["label"].iloc[0], "windows": len(d),
                     "accuracy": float(np.mean(d["pred"] == d["label"])),
                     "predicted": ", ".join(f"{c} {n}" for c, n in counts.items())})
    return pd.DataFrame(rows).sort_values(["accuracy", "session"], ignore_index=True)


def feature_importances(ds, cols):
    """RF importances (mean decrease in impurity) trained on all real windows."""
    real = ds[ds["copy"] == 0]
    rf = models.make_rf().fit(real[cols].to_numpy(), real["label"].to_numpy())
    return pd.Series(rf.feature_importances_, index=cols).sort_values(ascending=False)


def confused_pairs(y_true, y_pred):
    """[(true, predicted, fraction of the true class's windows)], worst first."""
    out = []
    for t in np.unique(y_true):
        mask = y_true == t
        for p in np.unique(y_pred[mask]):
            if p != t:
                out.append((t, p, float(np.mean(y_pred[mask] == p))))
    return sorted(out, key=lambda r: -r[2])


# ---- Stage 5: learning curve (is more data still helping?) --------------------------------

def learning_curve(ds, cols, seed=config.RANDOM_SEED):
    """Macro-F1 on held-out sessions vs. number of training sessions per class.

    For each grouped-CV fold, train on only 1, 2, 3, … sessions of each class
    and test on the fold's held-out sessions. If the curve is still rising at
    the right end, more recordings will help.
    """
    real = ds[ds["copy"] == 0].reset_index(drop=True)
    X, y, g = real[cols].to_numpy(), real["label"].to_numpy(), real["session"].to_numpy()
    rng = np.random.default_rng(seed)
    rows = []
    for fold, (train, test) in enumerate(models.group_splits(y, g)):
        by_class = {}
        for s in np.unique(g[train]):
            by_class.setdefault(y[g == s][0], []).append(s)
        max_n = max(len(v) for v in by_class.values())
        for n in range(1, max_n + 1):
            chosen = [s for v in by_class.values() for s in rng.permutation(v)[:n]]
            idx = np.flatnonzero(np.isin(g, chosen))
            m = models.make_rf().fit(X[idx], y[idx])
            rows.append({"fold": fold, "sessions_per_class": n,
                         "macro_f1": f1_score(y[test], m.predict(X[test]), average="macro")})
    lc = pd.DataFrame(rows)
    return lc.groupby("sessions_per_class")["macro_f1"].agg(["mean", "std", "count"]).reset_index()


# ---- Stage 6: feature selection + final Model 2 -------------------------------------------

def select_features(ds, cols, k_options=config.FEATURE_COUNT_OPTIONS, use_copies=False):
    """Pick how many features to keep (k) by grouped CV, then which ones.

    Inside each fold, features are ranked by RF importance on that fold's
    TRAINING data only (ranking on all data would leak the test sessions).
    Returns (chosen feature list, table of macro-F1 per k incl. 'all').
    """
    real = ds[ds["copy"] == 0].reset_index(drop=True)
    X, y, g = real[cols].to_numpy(), real["label"].to_numpy(), real["session"].to_numpy()
    copies = ds[ds["copy"] > 0]
    ks = [k for k in k_options if k < len(cols)] + [len(cols)]
    scores = {k: [] for k in ks}
    for train, test in models.group_splits(y, g):
        Xtr, ytr = X[train], y[train]
        if use_copies and len(copies):
            extra = copies[copies["session"].isin(set(g[train]))]
            Xtr = np.vstack([Xtr, extra[cols].to_numpy()])
            ytr = np.concatenate([ytr, extra["label"].to_numpy()])
        rank = np.argsort(-models.make_rf().fit(Xtr, ytr).feature_importances_)
        for k in ks:
            top = rank[:k]
            m = models.make_rf().fit(Xtr[:, top], ytr)
            scores[k].append(f1_score(y[test], m.predict(X[test][:, top]), average="macro"))
    table = pd.DataFrame({"k": [str(k) if k < len(cols) else f"all ({k})" for k in ks],
                          "n_features": ks,
                          "macro_f1": [np.mean(scores[k]) for k in ks],
                          "macro_f1_std": [np.std(scores[k]) for k in ks]})
    # Choose among the k_options only (the project brief: prune to ~10-20 features);
    # "all" stays in the table as a reference. Smallest k within 0.01 of the best
    # option: fewer features = faster live, less overfitting.
    options = table[table["n_features"] < len(cols)] if len(ks) > 1 else table
    best = options["macro_f1"].max()
    k = int(options[options["macro_f1"] >= best - 0.01]["n_features"].min())
    imp = feature_importances(ds, cols)
    return list(imp.index[:k]), table


def fit_final(model, ds, cols, use_copies=False):
    """Fit on ALL real windows (+ augmented copies if they helped in CV)."""
    part = ds if use_copies else ds[ds["copy"] == 0]
    return clone(model).fit(part[cols].to_numpy(), part["label"].to_numpy())


# ---- Stage 7: Model 1 (peak acceleration along one axis) ------------------------------------

MODEL1_FEATURES = {ax: f"acc_{ax}_peak" for ax in features.AXES}


class Model1Tree(DecisionTreeClassifier):
    """A normal decision tree on ONE feature, pruned after fitting so that no
    band of values is narrower than config.MODEL1_MIN_BAND_G.

    A too-narrow band is folded into its neighbour (that range then gets the
    neighbour's class). The pruning edits the fitted tree itself, so
    scikit-learn, the JSON export, the C export and cross-validation all use
    the same pruned tree.
    """

    def fit(self, X, y, sample_weight=None):
        super().fit(X, y, sample_weight)
        if self.n_features_in_ != 1:
            raise ValueError("Model1Tree is for a single feature")
        _prune_narrow(self.tree_, 0, -np.inf, np.inf, config.MODEL1_MIN_BAND_G)
        return self


def _prune_narrow(t, i, lo, hi, min_band):
    """Node i covers values in (lo, hi]. Replace it by one child while a side is too narrow."""
    while t.children_left[i] != -1:
        thr = t.threshold[i]
        if thr - lo < min_band:
            keep = t.children_right[i]          # left band too narrow: right side takes it over
        elif hi - thr < min_band:
            keep = t.children_left[i]
        else:
            break
        for arr in (t.children_left, t.children_right, t.feature, t.threshold, t.value):
            arr[i] = arr[keep]                  # node i becomes a copy of the kept child
    if t.children_left[i] != -1:
        thr = t.threshold[i]
        _prune_narrow(t, t.children_left[i], lo, thr, min_band)
        _prune_narrow(t, t.children_right[i], thr, hi, min_band)


def make_model1(depth):
    return Model1Tree(max_depth=depth, min_samples_leaf=3, class_weight="balanced",
                      random_state=config.RANDOM_SEED)


def evaluate_model1(ds, depths=config.MODEL1_DEPTHS):
    """Grouped CV for every (axis, depth). Returns (table, best axis, best depth, its CV result)."""
    rows, results = [], {}
    for ax, col in MODEL1_FEATURES.items():
        for d in depths:
            r = grouped_cv(make_model1(d), ds, [col])
            results[(ax, d)] = r
            rows.append({"axis": ax, "feature": col, "depth": d,
                         "macro_f1": r["fold_macro_f1"].mean(), "macro_f1_std": r["fold_macro_f1"].std(),
                         "accuracy": r["fold_accuracy"].mean()})
    table = pd.DataFrame(rows)
    # Best macro-F1; on a (near) tie prefer the shallower tree.
    best = table["macro_f1"].max()
    pick = table[table["macro_f1"] >= best - 0.01].sort_values(["depth", "macro_f1"], ascending=[True, False]).iloc[0]
    return table, pick["axis"], int(pick["depth"]), results[(pick["axis"], int(pick["depth"]))]


def separability(ds, col):
    """How much each pair of classes overlaps on one feature (0 = fully separate, 1 = identical range).

    Overlap = fraction of the two classes' windows that fall inside the
    range where both classes have their middle 90 % (5th-95th percentile).
    """
    real = ds[ds["copy"] == 0]
    labels = [c for c in config.CLASSES if c in set(real["label"])]
    vals = {c: real.loc[real["label"] == c, col].to_numpy() for c in labels}
    rows = []
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            lo = max(np.percentile(vals[a], 5), np.percentile(vals[b], 5))
            hi = min(np.percentile(vals[a], 95), np.percentile(vals[b], 95))
            both = np.concatenate([vals[a], vals[b]])
            frac = float(np.mean((both >= lo) & (both <= hi))) if hi > lo else 0.0
            rows.append({"class_a": a, "class_b": b, "overlap": frac})
    return pd.DataFrame(rows).sort_values("overlap", ascending=False, ignore_index=True)


# ---- Advice: what to record next ---------------------------------------------------------

def advice(counts, dropped, report, pairs, lc, target=config.TARGET_SESSIONS_PER_CLASS):
    """Plain-English list of what to record next, from the numbers."""
    tips = []
    for c in dropped:
        tips.append(f"Record '{c}': it has {counts[c]} usable recording(s), needs "
                    f"{config.MIN_SESSIONS_PER_CLASS} before it can be trained and tested at all.")
    for c in config.CLASSES:
        if c not in dropped and counts[c] < target:
            tips.append(f"Record more '{c}' ({counts[c]} so far, aim for {target}+).")
    for c, row in report.iterrows():
        if row["recall"] < config.WEAK_RECALL:
            worst = [p for p in pairs if p[0] == c][:1]
            also = f", mostly predicted as '{worst[0][1]}' ({worst[0][2]:.0%})" if worst else ""
            tips.append(f"'{c}' recall is only {row['recall']:.0%}{also}: record more varied '{c}' "
                        f"sessions (different speeds, days, strap positions)"
                        + (f", and more '{worst[0][1]}' too." if worst else "."))
    if len(lc) >= 2 and lc["mean"].iloc[-1] - lc["mean"].iloc[-2] > 0.02:
        tips.append("The learning curve is still rising: more sessions of every class should still help.")
    elif len(lc) >= 3:
        tips.append("The learning curve has flattened: more of the SAME kind of data won't help much; "
                    "add variety (other people, speeds, surfaces) instead.")
    return tips


def append_history(path, row):
    """One row per training run: watch the scores change as the dataset grows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False)


def now_str():
    return time.strftime("%Y-%m-%dT%H:%M:%S")
