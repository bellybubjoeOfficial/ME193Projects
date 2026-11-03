"""MediaPipe Pose (Tasks API) for the legs, plus a landmark cache.

mediapipe 0.10.35 on Python 3.14 has no mp.solutions, so this uses
PoseLandmarker in VIDEO mode (tracking between frames). The model file is
downloaded once into models/.

Landmarks come back as {name: (x_px, y_px, visibility)} for the leg points
below; names are the SUBJECT's left/right (left_* = leg "L").
"""

import urllib.request

import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks.python import BaseOptions, vision

from laptop.config import resolve

LANDMARKS = {
    "left_hip": 23, "right_hip": 24, "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28, "left_heel": 29, "right_heel": 30,
    "left_foot_index": 31, "right_foot_index": 32,
}
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_{v}/float16/latest/pose_landmarker_{v}.task")


def ensure_model(cfg):
    variant = cfg["pose"]["model"]
    path = resolve(cfg["pose"]["model_dir"]) / f"pose_landmarker_{variant}.task"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading MediaPipe pose model ({variant}) ...")
        urllib.request.urlretrieve(MODEL_URL.format(v=variant), path)
    return path


class PoseTracker:
    def __init__(self, cfg):
        p = cfg["pose"]
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model(cfg))),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=p["min_detection_confidence"],
            min_tracking_confidence=p["min_tracking_confidence"],
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._last_ms = -1

    def process(self, frame_bgr, t_ns):
        """Leg landmarks in pixels, or {} when no person is found."""
        h, w = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        # VIDEO mode needs strictly increasing timestamps.
        ms = max(int(t_ns // 1_000_000), self._last_ms + 1)
        self._last_ms = ms
        result = self._landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ms)
        if not result.pose_landmarks:
            return {}
        lms = result.pose_landmarks[0]
        return {name: (lms[i].x * w, lms[i].y * h, float(lms[i].visibility or 0.0))
                for name, i in LANDMARKS.items()}

    def close(self):
        self._landmarker.close()


# ---- cache: one row per frame, so thresholds can be re-tuned without re-running pose ----
def landmarks_row(frame_idx, t_ns, lm):
    row = {"frame_idx": frame_idx, "t_ns": t_ns}
    for name in LANDMARKS:
        x, y, v = lm.get(name, (np.nan, np.nan, 0.0))
        row[f"{name}_x"], row[f"{name}_y"], row[f"{name}_vis"] = x, y, v
    return row


def row_landmarks(row):
    if np.isnan(row["left_hip_x"]) and np.isnan(row["right_hip_x"]):
        return {}
    return {name: (row[f"{name}_x"], row[f"{name}_y"], row[f"{name}_vis"]) for name in LANDMARKS}


def save_landmarks(rows, path, model):
    df = pd.DataFrame(rows)
    df.attrs["model"] = model
    df.to_parquet(path, index=False)


def load_landmarks(path):
    return pd.read_parquet(path)
