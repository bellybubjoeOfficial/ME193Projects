#!/usr/bin/env python3
"""Stage 6: check the saved models load and agree, on THIS machine (laptop or Q).

    python scripts/check_models.py

Two model formats are saved by train.py:
  models/model2.joblib   scikit-learn object. Loads ONLY with the same
                         scikit-learn version that saved it (a different one
                         warns, or fails, or silently predicts differently).
  models/model*.json     plain numbers. The live App uses these: no
                         scikit-learn needed, any version-proof.

This prints the versions used for training vs. installed here, and checks
that the joblib and JSON versions of Model 2 predict the same classes.
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from wearable import config, forest_lite  # noqa: E402


def main():
    ok = True
    for name in config.LIVE_MODEL_FILES.values():
        path = config.MODELS_DIR / name
        if not path.exists():
            print(f"MISSING  {path}  (run scripts/train.py)")
            ok = False
            continue
        m = forest_lite.load(path)
        meta = m.get("meta", {})
        print(f"OK       {name}: {m['kind']}, {len(m['trees'])} tree(s), features {m['features'][:3]}"
              f"{'…' if len(m['features']) > 3 else ''}, trained {meta.get('trained_at')} "
              f"with scikit-learn {meta.get('sklearn')}")

    try:
        import joblib
        import sklearn
    except ImportError:
        print("\nscikit-learn/joblib not installed here: fine for the App (it uses the JSON files).")
        return 0 if ok else 1

    path = config.MODELS_DIR / "model2.joblib"
    if not path.exists():
        return 1
    print(f"\nInstalled here: scikit-learn {sklearn.__version__}, numpy {np.__version__}")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            saved = joblib.load(path)
        except Exception as e:  # noqa: BLE001 - any failure here means "version mismatch"
            print(f"FAIL     model2.joblib does not load: {e}")
            return 1
    trained_with = saved["meta"].get("sklearn")
    mismatch = [w for w in caught if "InconsistentVersion" in type(w.message).__name__]
    if trained_with != sklearn.__version__ or mismatch:
        print(f"WARN     model2.joblib was saved with scikit-learn {trained_with}, this is {sklearn.__version__}.\n"
              f"         Install the same one:  pip install scikit-learn=={trained_with}\n"
              f"         (or re-run scripts/train.py here). The JSON models are unaffected.")
        ok = False
    else:
        print(f"OK       model2.joblib: same scikit-learn version ({trained_with})")

    # Same predictions from both formats, on random feature vectors spread over each feature's thresholds.
    j = forest_lite.load(config.MODELS_DIR / "model2.json")
    thr = np.array([t for tree in j["trees"] for t in tree["threshold"]])
    X = np.random.default_rng(0).uniform(thr.min() - 1, thr.max() + 1, (2000, len(j["features"])))
    agree = float(np.mean(saved["model"].predict(X) == forest_lite.predict(j, X)))
    print(f"{'OK  ' if agree == 1 else 'FAIL'}     joblib vs JSON Model 2 agree on {agree:.1%} of 2000 test inputs")
    return 0 if ok and agree == 1 else 1


if __name__ == "__main__":
    sys.exit(main())
