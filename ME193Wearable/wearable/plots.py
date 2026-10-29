"""Stages 5-7 plots, saved as PNG files (no window needed: works over SSH)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from wearable import config  # noqa: E402


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def confusion(cm_table, title, path):
    """cm_table = models.confusion_table(...). Each row normalised to % of that true class."""
    cm = cm_table.to_numpy(dtype=float)
    pct = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    labels = [c.removeprefix("true_") for c in cm_table.index]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{pct[i, j]:.0%}\n({int(cm[i, j])})", ha="center", va="center",
                    color="white" if pct[i, j] > 0.6 else "black", fontsize=8)
    ax.set_xticks(range(len(labels)), [c.removeprefix("pred_") for c in cm_table.columns])
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=10)
    return _save(fig, path)


def importances(imp, path, top=20, chosen=()):
    """Horizontal bars of the top RF feature importances; chosen features highlighted."""
    imp = imp.iloc[:top][::-1]
    fig, ax = plt.subplots(figsize=(6, 0.28 * len(imp) + 1))
    ax.barh(imp.index, imp.values, color=["C0" if n in chosen else "lightgray" for n in imp.index])
    ax.set_xlabel("importance (mean decrease in impurity)")
    ax.set_title("Random Forest feature importances (blue = kept in Model 2)", fontsize=10)
    return _save(fig, path)


def learning_curve(lc, path):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.errorbar(lc["sessions_per_class"], lc["mean"], yerr=lc["std"].fillna(0), marker="o", capsize=3)
    ax.set_xlabel("training sessions per class")
    ax.set_ylabel("macro-F1 on held-out sessions")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(lc["sessions_per_class"])
    ax.grid(alpha=0.3)
    ax.set_title("Learning curve: still rising = more data helps", fontsize=10)
    return _save(fig, path)


def axis_distributions(ds, cols, path, best_col=None, thresholds=()):
    """One panel per axis: per-class box + points of the peak acceleration."""
    real = ds[ds["copy"] == 0]
    labels = [c for c in config.CLASSES if c in set(real["label"])]
    fig, axes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 3.8), sharey=True)
    rng = np.random.default_rng(0)
    for ax, col in zip(np.atleast_1d(axes), cols):
        data = [real.loc[real["label"] == c, col].to_numpy() for c in labels]
        ax.boxplot(data, showfliers=False)
        ax.set_xticks(range(1, len(labels) + 1), labels)
        for i, d in enumerate(data, 1):
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(d)), d, s=5, alpha=0.4)
        if col == best_col:
            for t in thresholds:
                ax.axhline(t, color="red", ls="--", lw=1)
            ax.set_title(f"{col}  (chosen; red = thresholds)", fontsize=10)
        else:
            ax.set_title(col, fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    np.atleast_1d(axes)[0].set_ylabel("peak |acceleration| in window (g, incl. gravity)")
    return _save(fig, path)
