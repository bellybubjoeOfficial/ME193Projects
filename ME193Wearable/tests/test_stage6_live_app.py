"""Stage 6 tests: live mode inside the App (python/main.py with a fake Bridge).

Uses one-leaf "models" (always predict one class) so the test is about the
App's plumbing: engaging a model, showing the prediction on the matrix, and
switching to a retrained model file by itself.
"""

import os

import pytest

from test_stage4_collection import app, drawn, feed  # noqa: F401  (pytest fixture + helpers)
from wearable import config, display, forest_lite, recording


def leaf_model(label, classes=config.CLASSES):
    """A 'forest' of one tree that is just a leaf voting for `label`."""
    value = [[1.0 if c == label else 0.0 for c in classes]]
    return {"kind": "model2", "classes": list(classes), "features": ["acc_y_peak"],
            "trees": [{"left": [-1], "right": [-1], "feature": [-2], "threshold": [-2.0], "value": value}],
            "meta": {"trained_at": "test"}}


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    d = tmp_path / "models"
    monkeypatch.setattr(config, "MODELS_DIR", d)
    forest_lite.save(leaf_model("sit"), d / "model2.json")
    return d


def run_live(main, seconds):
    for _ in range(int(seconds)):
        feed(main, 1.0)
        main.step_live()


def test_live_mode_shows_sit(app, models_dir):  # noqa: F811
    recording.write_live_request(config.CONTROL_DIR, "model2")
    assert app.step_live()
    run_live(app, 4)
    st = recording.read_status(config.CONTROL_DIR)
    assert st["state"] == "live" and st["label"] == "sit"
    assert drawn(app)[-1] == bytes(display.activity("sit", 1.0, 2))


def test_live_mode_reloads_a_retrained_model(app, models_dir):  # noqa: F811
    recording.write_live_request(config.CONTROL_DIR, "model2")
    app.step_live()
    run_live(app, 4)
    path = models_dir / "model2.json"
    forest_lite.save(leaf_model("airbike"), path)                 # train.py saved a new model
    st = os.stat(path)
    os.utime(path, (st.st_atime, st.st_mtime + 5))                # make sure the time differs
    run_live(app, 6)
    assert recording.read_status(config.CONTROL_DIR)["label"] == "airbike"
    assert any("retrained" in line for line in app.logger.lines)


def test_matrix_has_a_name_for_every_class():
    for c in config.CLASSES:
        text = display.ACTIVITY_TEXT[c]
        assert len(text) == 3 and all(ch in display.LETTERS for ch in text), c   # no '?' glyphs
    names = [display.ACTIVITY_TEXT[c] for c in config.CLASSES]
    assert len(set(names)) == len(names)
