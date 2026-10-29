"""Stage 6: save a trained tree model as plain JSON and run it with numpy only.

WHY: the App Lab container (where live inference runs) has numpy but no
scikit-learn, and a joblib/pickle file only loads with the SAME scikit-learn
version that saved it. A JSON file of the trees has neither problem: it is just
numbers, and predicting is a few lines of numpy below.

Works for a RandomForestClassifier (Model 2) and a DecisionTreeClassifier
(Model 1). The model dict:

    {"kind": "model2", "classes": [...], "features": [...],
     "trees": [{"left": [...], "right": [...], "feature": [...],
                "threshold": [...], "value": [[p_class0, p_class1, ...], ...]}, ...],
     "meta": {...anything: training date, CV scores, versions...}}

Node i is a leaf when left[i] == -1; otherwise go left if
x[feature[i]] <= threshold[i] (the same rule scikit-learn uses).
scikit-learn first rounds x to float32, so we do too: otherwise a value
within ~1e-7 of a threshold could take the other branch.
"""

import json
from pathlib import Path

import numpy as np


def _tree_dict(tree):
    t = tree.tree_
    value = t.value[:, 0, :]
    value = value / np.maximum(value.sum(axis=1, keepdims=True), 1e-12)   # leaf -> class probabilities
    return {
        "left": t.children_left.tolist(),
        "right": t.children_right.tolist(),
        "feature": t.feature.tolist(),
        "threshold": t.threshold.tolist(),
        "value": np.round(value, 6).tolist(),
    }


def export(model, feature_names, kind, meta=None):
    """sklearn tree/forest -> JSON-able dict. feature_names = the columns it was trained on."""
    trees = getattr(model, "estimators_", [model])          # a forest has many, a tree is just one
    if len(feature_names) != model.n_features_in_:
        raise ValueError(f"model has {model.n_features_in_} features, got {len(feature_names)} names")
    return {
        "kind": kind,
        "classes": [str(c) for c in model.classes_],
        "features": list(feature_names),
        "trees": [_tree_dict(t) for t in trees],
        "meta": meta or {},
    }


def save(model_dict, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(model_dict))
    tmp.replace(path)           # atomic: the App never loads half a file


def load(path):
    return json.loads(Path(path).read_text())


def _leaf_values(tree, X):
    """Class-probability row of the leaf each sample in X lands in."""
    left, right = np.asarray(tree["left"]), np.asarray(tree["right"])
    feat, thr = np.asarray(tree["feature"]), np.asarray(tree["threshold"])
    node = np.zeros(len(X), dtype=int)
    rows = np.arange(len(X))
    while True:
        inner = left[node] != -1
        if not inner.any():
            break
        n = node[inner]
        go_left = X[rows[inner], feat[n]] <= thr[n]
        node[inner] = np.where(go_left, left[n], right[n])
    return np.asarray(tree["value"])[node]


def predict_proba(model_dict, X):
    """Average class probabilities over all trees (what RandomForest does). X: (n, n_features)."""
    X = np.atleast_2d(np.asarray(X, dtype=np.float32))     # same rounding as scikit-learn
    return np.mean([_leaf_values(t, X) for t in model_dict["trees"]], axis=0)


def predict(model_dict, X):
    classes = np.asarray(model_dict["classes"])
    return classes[np.argmax(predict_proba(model_dict, X), axis=1)]


def features_from_row(model_dict, feats):
    """Pick the model's features, in its order, from a {name: value} dict."""
    return np.array([[feats[name] for name in model_dict["features"]]])


def _subtree_class(t, i):
    """Class index if every leaf under node i predicts the same class, else None."""
    if t["left"][i] == -1:
        return int(np.argmax(t["value"][i]))
    a, b = _subtree_class(t, t["left"][i]), _subtree_class(t, t["right"][i])
    return a if a is not None and a == b else None


def rules_text(model_dict, units="g"):
    """Human-readable if/else rules for a single tree (Model 1).

    Splits whose two sides predict the same class (sklearn keeps them because
    they make the leaves "purer") are printed as one rule.
    """
    if len(model_dict["trees"]) != 1:
        raise ValueError("rules_text is for a single decision tree")
    t = model_dict["trees"][0]
    classes, names = model_dict["classes"], model_dict["features"]
    lines = []

    def walk(i, depth):
        pad = "    " * depth
        k = _subtree_class(t, i)
        if k is not None:
            # value = class mix of the training windows reaching this node
            # (after class balancing, so every class counts equally).
            lines.append(f"{pad}-> {classes[k]}  ({t['value'][i][k]:.0%} of balanced training windows here)")
            return
        name, thr = names[t["feature"][i]], t["threshold"][i]
        lines.append(f"{pad}if {name} <= {thr:.3f} {units}:")
        walk(t["left"][i], depth + 1)
        lines.append(f"{pad}else:  # {name} > {thr:.3f} {units}")
        walk(t["right"][i], depth + 1)

    walk(0, 0)
    return "\n".join(lines)


def interval_rules(model_dict, units="g"):
    """For a ONE-feature tree: the number line cut into ranges, one class each.

    [(low, high, class), ...] with low = -inf for the first range and
    high = +inf for the last; neighbouring ranges with the same class are merged.
    """
    if len(model_dict["features"]) != 1 or len(model_dict["trees"]) != 1:
        raise ValueError("interval_rules is for a single tree on a single feature")
    t = model_dict["trees"][0]
    cuts, todo = set(), [0]
    while todo:                         # thresholds of the nodes reachable from the root
        i = todo.pop()
        if t["left"][i] != -1:
            cuts.add(t["threshold"][i])
            todo += [t["left"][i], t["right"][i]]
    cuts = sorted(cuts)
    edges = [-np.inf] + cuts + [np.inf]
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        # Any point inside (lo, hi] lands in the same leaf; test the middle (away from float rounding).
        if not np.isfinite(lo):
            probe = hi - 1.0
        elif not np.isfinite(hi):
            probe = lo + 1.0
        else:
            probe = (lo + hi) / 2
        label = str(predict(model_dict, [[probe]])[0])
        if out and out[-1][2] == label:
            out[-1] = (out[-1][0], hi, label)
        else:
            out.append((lo, hi, label))
    return out


def interval_text(model_dict, units="g"):
    name = model_dict["features"][0]
    lines = []
    for lo, hi, label in interval_rules(model_dict):
        if not np.isfinite(lo):
            rng = f"{name} <= {hi:.2f} {units}"
        elif not np.isfinite(hi):
            rng = f"{name} >  {lo:.2f} {units}"
        else:
            rng = f"{lo:.2f} < {name} <= {hi:.2f} {units}"
        lines.append(f"  {rng:34s} -> {label}")
    return "\n".join(lines)


# ---- Export to C (optional: to run a model on the MCU) ------------------------

def _float_below(thr):
    """Largest float32 <= thr. For a float x, (x <= this) is exactly (x <= thr),
    so the C code (all float) branches like scikit-learn (float32 x vs double thr)."""
    f = np.float32(thr)
    return float(f if f <= thr else np.nextafter(f, np.float32(-np.inf)))


def _c_tree(t, n_classes, name):
    """One tree as a C function that adds its class probabilities to p[]."""
    lines = [f"static void {name}(const float *x, float *p) {{"]

    def walk(i, depth):
        pad = "    " * depth
        if t["left"][i] == -1:
            adds = " ".join(f"p[{k}] += {v:.6g}f;" for k, v in enumerate(t["value"][i]) if v > 0)
            lines.append(f"{pad}{adds}")
            return
        lines.append(f"{pad}if (x[{t['feature'][i]}] <= {_float_below(t['threshold'][i]):.9g}f) {{")
        walk(t["left"][i], depth + 1)
        lines.append(f"{pad}}} else {{")
        walk(t["right"][i], depth + 1)
        lines.append(f"{pad}}}")

    walk(0, 1)
    lines.append("}")
    return "\n".join(lines)


def to_c(model_dict, prefix=None):
    """The model as a self-contained C header (no libraries, no malloc).

    int <prefix>_predict(const float *x) returns the class index; x holds the
    features in the order listed at the top. Same rule as above
    (x <= threshold -> left), so it predicts exactly what predict() does.
    """
    prefix = prefix or model_dict.get("kind", "model")
    classes, feats, trees = model_dict["classes"], model_dict["features"], model_dict["trees"]
    k = len(classes)
    out = [f"// {prefix}: generated by wearable/forest_lite.py from {prefix}.json. Do not edit.",
           f"// {len(trees)} tree(s), {len(feats)} feature(s). x[] must hold, in this order:"]
    out += [f"//   x[{i}] = {f}" for i, f in enumerate(feats)]
    out += [f"#pragma once", f"#define {prefix.upper()}_N_FEATURES {len(feats)}",
            f"#define {prefix.upper()}_N_CLASSES {k}",
            f"static const char *const {prefix}_classes[{k}] = {{"
            + ", ".join(f'"{c}"' for c in classes) + "};", ""]
    out += [_c_tree(t, k, f"{prefix}_tree{j}") for j, t in enumerate(trees)]
    body = [f"int {prefix}_predict(const float *x) {{", f"    float p[{k}] = {{0}};"]
    body += [f"    {prefix}_tree{j}(x, p);" for j in range(len(trees))]
    body += ["    int best = 0;", f"    for (int i = 1; i < {k}; i++) if (p[i] > p[best]) best = i;",
             "    return best;", "}"]
    return "\n".join(out + [""] + body) + "\n"
