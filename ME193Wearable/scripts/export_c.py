#!/usr/bin/env python3
"""Stage 6 (optional): export the trained models as C headers, to run them on the MCU.

    python scripts/export_c.py              # -> models/model1.h, models/model2.h

Each header has one function, e.g. `int model1_predict(const float *x)`,
returning an index into `model1_classes[]`. No libraries, no malloc.

The tree part is the easy half. The MCU would also have to compute the
FEATURES exactly like wearable/features.py:
  * Model 1: one feature, the peak |accel| on one axis over the last 256
    samples. A few lines of C; this one really is practical on the MCU.
  * Model 2: 10-20 features incl. FFT bands, dominant frequency and a gravity
    filter. Doable, but a lot of C to write and test, which is why this
    project runs Model 2 on the Linux side.

Why not m2cgen / micromlgen? They work too (`pip install m2cgen`, then
`m2cgen.export_to_c(joblib.load('models/model2.joblib')['model'])`), but they
read the scikit-learn object, so they need the training scikit-learn version
(m2cgen's last release is from 2022). This reads the JSON file the App already
uses, so the C code is guaranteed to be the same model the App runs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wearable import config, forest_lite  # noqa: E402


def main():
    for name in config.LIVE_MODEL_FILES:
        src = config.MODELS_DIR / config.LIVE_MODEL_FILES[name]
        if not src.exists():
            print(f"skip {name}: {src} missing (run scripts/train.py)")
            continue
        dst = src.with_suffix(".h")
        dst.write_text(forest_lite.to_c(forest_lite.load(src), name))
        print(f"{dst}  ({dst.stat().st_size / 1024:.0f} KB of source)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
