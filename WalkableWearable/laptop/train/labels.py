"""Stage 6: heel-strike labels for a synced session, on the IMU's clock.

Source (in order of preference):
  hand    truth.csv in the session folder (mark it with
          python -m laptop.vision.handcheck --clip data/recordings/<folder>)
  vision  the Stage 3 detector run on the session video (heavy pose model)
Video times are moved onto the IMU timeline with the Stage 4 stomp
correction: t_video = t_imu + r(t)  ->  t_imu = t_video - r(t_video).

Also returns which video frames the camera could actually see a heel in, so
training ignores stretches where a missing label just means "not visible".

Writes labels.csv: t_imu_ns, leg, frame_idx, source, confidence.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from laptop.config import load_config
from laptop.sync.stomp import correction
from laptop.vision.clip import Clip
from laptop.vision.pose import load_landmarks


def video_to_imu(folder):
    """Function t_video_ns -> t_imu_ns from sync_correction.json (identity if absent)."""
    path = Path(folder) / "sync_correction.json"
    if not path.exists():
        return lambda t: np.asarray(t, dtype=np.int64), False
    fwd = correction(json.load(open(path)))
    # r(t) changes by ~ms per minute, so evaluating it at the video time is exact enough.
    return lambda t: np.asarray(np.asarray(t) - (fwd(t) - np.asarray(t)), dtype=np.int64), True


def session_labels(folder, cfg=None, relabel=False, verbose=True):
    """Returns (labels DataFrame, video frame times on IMU clock, heel-visible per frame)."""
    cfg = cfg or load_config()
    folder = Path(folder)
    clip = Clip(folder)
    to_imu, corrected = video_to_imu(folder)
    lm_path = clip.landmarks_path(cfg["pose"]["model"])
    out_path = folder / "labels.csv"

    if (folder / "truth.csv").exists():
        truth = pd.read_csv(folder / "truth.csv")
        labels = pd.DataFrame({"t_imu_ns": to_imu(truth["t_ns"].to_numpy()), "leg": truth["leg"],
                               "frame_idx": truth["frame_idx"], "source": "hand", "confidence": 1.0})
    else:
        if relabel or not (folder / "steps.csv").exists() or not lm_path.exists():
            from laptop.vision.step_viewer import detect_clip
            if verbose:
                print(f"  vision labels for {folder.name} (pose on every frame, ~19 fps) ...")
            detect_clip(clip, cfg, display=False)
        steps = pd.read_csv(folder / "steps.csv")
        labels = pd.DataFrame({"t_imu_ns": to_imu(steps["t_ns"].to_numpy()), "leg": steps["leg"],
                               "frame_idx": steps["frame_idx"], "source": "vision",
                               "confidence": steps["confidence"]})
    labels.to_csv(out_path, index=False)

    vmin = cfg["steps"]["visibility_min"]
    if lm_path.exists():
        lm = load_landmarks(lm_path)
        seen = ((lm["left_heel_vis"].fillna(0) >= vmin) | (lm["right_heel_vis"].fillna(0) >= vmin)).to_numpy()
        frame_t = to_imu(lm["t_ns"].to_numpy())
    else:  # hand labels without a pose run: trust the whole video
        frame_t = to_imu(clip.t_ns)
        seen = np.ones(len(frame_t), dtype=bool)
    if verbose and not corrected:
        print(f"  WARNING {folder.name}: no sync_correction.json -- labels use the raw clocks")
    return labels, frame_t, seen
