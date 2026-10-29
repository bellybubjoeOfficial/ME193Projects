#!/usr/bin/env python3
"""Stages 5-7 in one command: evaluate on all real data, train + save both models.

    python scripts/train.py                 # everything (on the Q or the laptop)
    python scripts/train.py --no-augment    # skip the augmentation comparison (faster)

Re-run it after every batch of new recordings: it always uses EVERY recording
in data/sessions.csv, so the models get better as you add data. Only new
recordings need their features computed (the rest come from .feature_cache/).

Outputs:
  reports/latest/report.md    the full Stage 5-7 report (tables + advice)
  reports/latest/*.png        confusion matrices, importances, learning curve, Model 1 axes
  reports/history.csv         one row per run: watch the scores improve with more data
  models/model2.json          Model 2 (Random Forest, top features), used by the live App
  models/model2.joblib        same model for scikit-learn (laptop analysis)
  models/model1.json          Model 1 (one-axis peak, shallow tree), used by the live App

On the Q it runs at the lowest CPU priority and waits while the App is
counting down or recording, so it never costs you samples.
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import sklearn  # noqa: E402

from wearable import config, features, forest_lite, models, plots, recording, training  # noqa: E402

PAIR_WHY = {
    frozenset({"sit", "stand"}): "both are still, so the peak is just gravity; it only differs if the shin tilts when you sit",
    frozenset({"sit", "airbike"}): "pedalling in the air is smooth (no heel strike), so the peak stays close to the still value",
    frozenset({"stand", "airbike"}): "pedalling is smooth (no heel strike), so the peak stays near 1 g, like standing",
    frozenset({"walk", "run"}): "a slow jog and a brisk walk jolt the shin about equally hard",
    frozenset({"walk", "airbike"}): "a gentle walk and hard pedalling can reach similar peaks",
    frozenset({"stand", "walk"}): "very slow walking barely jolts the shin",
    frozenset({"sit", "walk"}): "very slow walking barely jolts the shin",
    frozenset({"run", "airbike"}): "a slow jog's heel strike and a hard pedal stroke can jolt the shin equally",
    frozenset({"stand", "run"}): "pauses inside a run recording (waiting, turning) look like standing",
    frozenset({"sit", "run"}): "pauses inside a run recording look like being still",
}


def wait_until_app_idle():
    """Don't compete with a recording for CPU: wait while the App counts down or records."""
    warned = False
    while (recording.read_status(config.CONTROL_DIR) or {}).get("state") in ("countdown", "recording"):
        if not warned:
            print("The App is recording: waiting for it to finish before training…", flush=True)
            warned = True
        time.sleep(3)


def fmt(df, floats="{:.3f}"):
    return "```\n" + df.to_string(float_format=floats.format) + "\n```"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-augment", action="store_true", help="skip augmented copies")
    ap.add_argument("--no-cache", action="store_true", help="recompute all features")
    ap.add_argument("--out", type=Path, default=config.REPORTS_DIR / "latest")
    ap.add_argument("--models", type=Path, default=config.MODELS_DIR)
    args = ap.parse_args()
    try:
        os.nice(19)            # lowest CPU priority: the App always wins
    except OSError:
        pass
    wait_until_app_idle()
    t0 = time.time()
    out, mdir = args.out, args.models
    out.mkdir(parents=True, exist_ok=True)
    md = [f"# Training report ({training.now_str()})", ""]

    # ---- Load ---------------------------------------------------------------
    recs, notes = training.load_real_recordings()
    copies = 0 if args.no_augment else config.AUGMENT_COPIES
    print(f"{len(recs)} usable recordings; computing features (cached ones are reused)…", flush=True)
    ds_all = training.build_feature_table(recs, copies, None if args.no_cache else config.FEATURE_CACHE_DIR)
    if notes:
        md += ["## Data notes", ""] + [f"- {n}" for n in notes] + [""]
        print("\n".join("  " + n for n in notes))
    try:
        ds, counts, dropped = training.check_enough_data(ds_all)
    except training.NotEnoughData as e:
        md += ["## Not enough data yet", "", str(e), "",
               recording.format_summary(recording.read_sessions(config.SESSIONS_CSV))]
        (out / "report.md").write_text("\n".join(md))
        print(f"\nNOT ENOUGH DATA YET: {e}\nRecord more with scripts/collect.py, then run this again.")
        return 2

    cols = features.feature_names()
    real = ds[ds["copy"] == 0]
    X, y, g = real[cols].to_numpy(), real["label"].to_numpy(), real["session"].to_numpy()
    sess = real.groupby("label")["session"].nunique()
    md += ["## Dataset", "", fmt(real.groupby("label").agg(sessions=("session", "nunique"),
                                                           windows=("session", "size"))), ""]
    if dropped:
        md += [f"Not trained yet (fewer than {config.MIN_SESSIONS_PER_CLASS} recordings): "
               + ", ".join(f"{c} ({counts[c]})" for c in dropped), ""]
    print(f"Classes: { {k: int(v) for k, v in sess.items()} }  ({len(real)} windows)" + (f"; not enough data for {dropped}" if dropped else ""))

    # ---- Stage 5: compare models ------------------------------------------------
    print("Stage 5: grouped cross-validation of 5 models…", flush=True)
    table, results = models.compare_models(X, y, g)
    rf = results["random_forest"]
    report = training.per_class_report(y, rf["y_pred"])
    cm = models.confusion_table(y, rf["y_pred"])
    pairs = training.confused_pairs(y, rf["y_pred"])
    imp = training.feature_importances(ds, cols)
    lc = training.learning_curve(ds, cols)
    plain_f1 = float(rf["fold_macro_f1"].mean())

    use_copies, aug_line = False, "Augmentation skipped (--no-augment)."
    if copies:
        aug = training.grouped_cv(models.make_rf(), ds, cols, use_copies=True)
        aug_f1 = float(aug["fold_macro_f1"].mean())
        use_copies = aug_f1 > plain_f1 + 0.005
        aug_line = (f"Random Forest macro-F1 without augmentation {plain_f1:.3f}, with {copies} augmented "
                    f"copies per training recording {aug_f1:.3f} -> "
                    + ("augmentation HELPS, used for the final model." if use_copies
                       else "no clear gain, final model trained on real data only."))

    md += ["## Stage 5: model comparison (session-grouped CV)", "", fmt(table), "",
           "### Random Forest per class", "", fmt(report), "",
           "![RF confusion](confusion_model2_all_features.png)", "", "### Most confused", ""]
    md += [f"- true **{t}** predicted as **{p}**: {f:.0%} of {t} windows" for t, p, f in pairs[:5]] or ["- none"]
    sessions_table = training.per_session_table(ds, rf["y_pred"])
    md += ["", "### Per recording (worst first)", "",
           "Each recording scored while it was held out. A recording mostly predicted as another "
           "class is a hard case or a bad/mislabelled file: look at its plot first.", "",
           fmt(sessions_table, "{:.2f}"), ""]
    md += ["", "### Augmentation", "", aug_line, "",
           "### Learning curve", "", fmt(lc), "", "![learning curve](learning_curve.png)", "",
           "### Feature importances (top 20)", "", fmt(imp.iloc[:20].to_frame("importance")), ""]

    # ---- Stage 6: Model 2 ----------------------------------------------------------
    print("Stage 6: choosing features + training Model 2…", flush=True)
    chosen, ktable = training.select_features(ds, cols, use_copies=use_copies)
    m2_cv = training.grouped_cv(models.make_rf(), ds, chosen, use_copies=use_copies)
    m2 = training.fit_final(models.make_rf(), ds, chosen, use_copies=use_copies)
    m2_f1 = float(m2_cv["fold_macro_f1"].mean())
    meta = {"trained_at": training.now_str(), "sessions": {k: int(v) for k, v in sess.items()},
            "windows": int(len(real)), "augmented": use_copies, "cv_macro_f1": round(m2_f1, 4),
            "sklearn": sklearn.__version__, "numpy": np.__version__}
    mdir.mkdir(parents=True, exist_ok=True)
    forest_lite.save(forest_lite.export(m2, chosen, "model2", meta), mdir / "model2.json")
    joblib.dump({"model": m2, "features": chosen, "meta": meta}, mdir / "model2.joblib")
    cm2 = models.confusion_table(y, m2_cv["y_pred"])
    md += ["## Stage 6: Model 2 (all IMU outputs)", "", "How many features to keep (grouped CV):", "",
           fmt(ktable), "", f"Kept {len(chosen)} features: " + ", ".join(f"`{c}`" for c in chosen), "",
           f"CV macro-F1 with these features: **{m2_f1:.3f}** "
           f"(accuracy {m2_cv['fold_accuracy'].mean():.3f}). The feature ranking used all data, so this "
           "is a little optimistic; the 'k' table above is the honest estimate.", "",
           fmt(training.per_class_report(y, m2_cv["y_pred"])), "",
           "![Model 2 confusion](confusion_model2.png)", "",
           "![importances](importances.png)", ""]

    # ---- Stage 7: Model 1 -----------------------------------------------------------
    print("Stage 7: Model 1 (one-axis peak acceleration)…", flush=True)
    m1_table, axis, depth, m1_cv = training.evaluate_model1(ds)
    col = training.MODEL1_FEATURES[axis]
    m1 = training.fit_final(training.make_model1(depth), ds, [col])
    m1_f1 = float(m1_cv["fold_macro_f1"].mean())
    m1_dict = forest_lite.export(m1, [col], "model1", {**meta, "cv_macro_f1": round(m1_f1, 4), "axis": axis,
                                                        "depth": depth, "augmented": False})
    forest_lite.save(m1_dict, mdir / "model1.json")
    t = m1.tree_
    thresholds = sorted(float(v) for v, l in zip(t.threshold, t.children_left) if l != -1)
    rules = forest_lite.rules_text(m1_dict) + "\n\nSame thing as ranges:\n" + forest_lite.interval_text(m1_dict)
    sep = training.separability(ds, col)
    m1_pairs = training.confused_pairs(y, m1_cv["y_pred"])
    cm1 = models.confusion_table(y, m1_cv["y_pred"])
    why = []
    for _, r in sep.iterrows():
        if r["overlap"] > 0.2:
            reason = PAIR_WHY.get(frozenset({r["class_a"], r["class_b"]}), "their peaks fall in the same range")
            why.append(f"- **{r['class_a']} vs {r['class_b']}**: {r['overlap']:.0%} of their windows fall in "
                       f"the shared range: {reason}. One peak value measures how hard the shin is jolted, "
                       "not its rhythm, rotation or posture.")
    md += ["## Stage 7: Model 1 (peak acceleration, ONE axis)", "",
           "Every axis x depth, grouped CV:", "", fmt(m1_table), "",
           f"Best: **{axis} axis** (`{col}`), depth {depth}, macro-F1 **{m1_f1:.3f}** "
           f"vs Model 2 **{m2_f1:.3f}**.", "", "Rules:", "", "```", rules, "```", "",
           fmt(training.per_class_report(y, m1_cv["y_pred"])), "",
           "![Model 1 confusion](confusion_model1.png)", "", "![axes](model1_axes.png)", "",
           "### What Model 1 can't separate", ""] + (why or ["- every pair separates well on this axis"]) + [""]

    # ---- Advice, history, plots --------------------------------------------------------
    tips = training.advice(counts, dropped, report, pairs, lc)
    md += ["## What to record next", ""] + [f"- {s}" for s in tips or ["Nothing urgent: data looks sufficient."]]
    plots.confusion(cm, "Random Forest, all features (CV)", out / "confusion_model2_all_features.png")
    plots.confusion(cm2, f"Model 2: RF, {len(chosen)} features (CV)", out / "confusion_model2.png")
    plots.confusion(cm1, f"Model 1: peak {axis}, depth {depth} (CV)", out / "confusion_model1.png")
    plots.importances(imp, out / "importances.png", chosen=chosen)
    plots.learning_curve(lc, out / "learning_curve.png")
    plots.axis_distributions(ds, list(training.MODEL1_FEATURES.values()), out / "model1_axes.png",
                             best_col=col, thresholds=thresholds)
    (out / "report.md").write_text("\n".join(md) + "\n")
    training.append_history(config.REPORTS_DIR / "history.csv", {
        "trained_at": meta["trained_at"], **{f"sessions_{c}": int(counts[c]) for c in config.CLASSES},
        "windows": len(real), "rf_all_macro_f1": round(plain_f1, 4), "model2_macro_f1": round(m2_f1, 4),
        "model2_features": len(chosen), "augmented": use_copies,
        "model1_axis": axis, "model1_depth": depth, "model1_macro_f1": round(m1_f1, 4)})

    # ---- Console summary --------------------------------------------------------------------
    print(f"\n{table.to_string(float_format='{:.3f}'.format)}\n")
    print("Per recording (Random Forest, held out):\n" + sessions_table.to_string(float_format="{:.2f}".format) + "\n")
    print(f"Model 2: {len(chosen)} features, CV macro-F1 {m2_f1:.3f}" + ("  (augmented)" if use_copies else ""))
    print(f"Model 1: peak along {axis}, depth {depth}, CV macro-F1 {m1_f1:.3f}\n{rules}\n")
    print("What to record next:\n" + "\n".join("  - " + s for s in tips or ["nothing urgent"]))
    print(f"\nReport: {out / 'report.md'}\nModels: {mdir}/model1.json, model2.json  ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
