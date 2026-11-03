"""A recorded clip: either a Stage 2 folder or any video file.

    data/clips/<name>/          video.mp4 + frames.csv (laptop timestamps)
    some_phone_video.mov        timestamps = frame_idx / fps; outputs go in
                                some_phone_video_steps/

Outputs written next to the clip: landmarks_<model>.parquet, steps.csv, truth.csv,
annotated.mp4.
"""

from pathlib import Path

import cv2
import pandas as pd


class Clip:
    def __init__(self, path):
        p = Path(path)
        if p.is_dir():
            self.dir, self.video = p, p / "video.mp4"
            frames_csv = p / "frames.csv"
        else:
            self.video = p
            self.dir = p.with_name(p.stem + "_steps")
            frames_csv = None
        if not self.video.exists():
            raise FileNotFoundError(f"{self.video} -- give the full clip folder name "
                                    "(type data/clips/2 and press Tab)")
        self.dir.mkdir(exist_ok=True)
        cap = cv2.VideoCapture(str(self.video))
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.n_frames_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        self.t_ns = (pd.read_csv(frames_csv)["t_ns"].to_numpy()
                     if frames_csv and frames_csv.exists() else None)
        self.name = p.name

    def path(self, name):
        return self.dir / name

    def landmarks_path(self, model):
        """Landmark cache per pose model, so switching model never reuses stale data."""
        return self.dir / f"landmarks_{model}.parquet"

    def time_of(self, frame_idx):
        if self.t_ns is not None and frame_idx < len(self.t_ns):
            return int(self.t_ns[frame_idx])
        return int(frame_idx * 1e9 / self.fps)

    def frames(self):
        """Yield (frame_idx, t_ns, frame) for every frame in order."""
        cap = cv2.VideoCapture(str(self.video))
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield idx, self.time_of(idx), frame
            idx += 1
        cap.release()
