"""Stage 3 hand-check: step through a clip, mark the TRUE heel strikes.

    python -m laptop.vision.handcheck --clip data/clips/<folder>

Keys
    d / →        next frame          a / ←        previous frame
    D / ]        +10 frames          A / [        -10 frames
    space        play / pause        n / p        jump to next / previous detection
    l            mark LEFT heel strike on this frame (press again to unmark)
    r            mark RIGHT heel strike on this frame
    x            delete any mark within 2 frames
    s            save                q / Esc      save, quit, print the report

Mark the first frame where the heel touches the ground. Marks are saved to
truth.csv (frame_idx, leg, t_ns). If steps.csv exists (run step_viewer
first) its detections are shown on the timeline and compared on quit.
"""

import argparse
from collections import OrderedDict

import cv2
import pandas as pd

from laptop.config import load_config
from laptop.vision.clip import Clip
from laptop.vision.compare import compare, format_report
from laptop.vision.overlay import draw_legs, leg_colors
from laptop.vision.pose import load_landmarks, row_landmarks

WINDOW = "Hand-check (l/r mark, a/d step, q quit)"
LEFT_KEYS = {ord("a"), 63234, 2424832, 65361}
RIGHT_KEYS = {ord("d"), 63235, 2555904, 65363}


class FrameReader:
    """Random access with a small cache; sequential reads avoid seeking."""

    def __init__(self, path, cache=200):
        self.cap = cv2.VideoCapture(str(path))
        self.n = 0
        while self.cap.grab():  # exact count (container metadata can be off)
            self.n += 1
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.next_idx = 0
        self.cache = OrderedDict()
        self.max_cache = cache

    def get(self, idx):
        if idx in self.cache:
            self.cache.move_to_end(idx)
            return self.cache[idx]
        if idx != self.next_idx:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        self.next_idx = idx + 1
        if not ok:
            return None
        self.cache[idx] = frame
        if len(self.cache) > self.max_cache:
            self.cache.popitem(last=False)
        return frame


def load_truth(path):
    """Set of (frame_idx, leg)."""
    if not path.exists():
        return set()
    return {(int(r.frame_idx), r.leg) for r in pd.read_csv(path).itertuples()}


def save_truth(marks, clip):
    rows = [{"frame_idx": f, "leg": leg, "t_ns": clip.time_of(f)} for f, leg in sorted(marks)]
    pd.DataFrame(rows, columns=["frame_idx", "leg", "t_ns"]).to_csv(clip.path("truth.csv"), index=False)


def draw_timeline(img, idx, n, marks, det, colors):
    h, w = img.shape[:2]
    y0 = h - 50
    cv2.rectangle(img, (0, y0), (w, h), (30, 30, 30), -1)
    xs = lambda f: int(10 + (w - 20) * f / max(n - 1, 1))
    for f, leg in det:          # detections: short colored ticks on top
        cv2.line(img, (xs(f), y0 + 5), (xs(f), y0 + 20), colors[leg], 2)
    for f, leg in marks:        # truth: tall white ticks with leg letter
        cv2.line(img, (xs(f), y0 + 22), (xs(f), h - 5), (255, 255, 255), 2)
    cv2.line(img, (xs(idx), y0), (xs(idx), h), (0, 255, 255), 2)


def run(clip, cfg):
    reader = FrameReader(clip.video)
    n = reader.n
    marks = load_truth(clip.path("truth.csv"))
    det = []
    if clip.path("steps.csv").exists():
        sdf = pd.read_csv(clip.path("steps.csv"))
        det = [(int(r.frame_idx), r.leg) for r in sdf.itertuples()]
    lm_path = clip.landmarks_path(cfg["pose"]["model"])
    lmdf = load_landmarks(lm_path) if lm_path.exists() else None
    colors = leg_colors(cfg)
    idx, playing = 0, False
    print(f"{clip.name}: {n} frames, {len(det)} detections, {len(marks)} existing marks")

    while True:
        frame = reader.get(idx)
        if frame is None:
            idx = max(0, idx - 1)
            playing = False
            continue
        img = frame.copy()
        if lmdf is not None and idx < len(lmdf):
            draw_legs(img, row_landmarks(lmdf.iloc[idx]), cfg)
        here = [leg for f, leg in marks if f == idx]
        near_det = [(f - idx, leg) for f, leg in det if abs(f - idx) <= 3]
        lines = [f"frame {idx}/{n - 1}   t {(clip.time_of(idx) - clip.time_of(0)) / 1e9:.3f} s",
                 f"marks {len(marks)}   detections {len(det)}"]
        if near_det:
            lines.append("detector: " + "  ".join(f"{leg}@{d:+d}" for d, leg in near_det))
        for i, line in enumerate(lines):
            cv2.putText(img, line, (12, 32 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
            cv2.putText(img, line, (12, 32 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        for k, leg in enumerate(here):
            cv2.putText(img, f"TRUE {leg}", (img.shape[1] - 260, 70 + 60 * k),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, colors[leg], 5)
        draw_timeline(img, idx, n, marks, det, colors)
        cv2.imshow(WINDOW, img)

        key = cv2.waitKeyEx(int(1000 / clip.fps) if playing else 0)
        if key == -1:
            if playing:
                idx = min(n - 1, idx + 1)
            continue
        ch = key & 0xFF if key < 256 else None
        if key in RIGHT_KEYS:
            idx, playing = min(n - 1, idx + 1), False
        elif key in LEFT_KEYS:
            idx, playing = max(0, idx - 1), False
        elif ch in (ord("D"), ord("]")):
            idx = min(n - 1, idx + 10)
        elif ch in (ord("A"), ord("[")):
            idx = max(0, idx - 10)
        elif ch == ord(" "):
            playing = not playing
        elif ch == ord("n"):
            later = [f for f, _ in det if f > idx]
            idx = min(later) if later else idx
        elif ch == ord("p"):
            earlier = [f for f, _ in det if f < idx]
            idx = max(earlier) if earlier else idx
        elif ch in (ord("l"), ord("r")):
            marks ^= {(idx, "L" if ch == ord("l") else "R")}  # toggle
            save_truth(marks, clip)
        elif ch == ord("x"):
            marks -= {m for m in marks if abs(m[0] - idx) <= 2}
            save_truth(marks, clip)
        elif ch == ord("s"):
            save_truth(marks, clip)
            print(f"saved {len(marks)} marks")
        elif ch in (ord("q"), 27):
            break

    save_truth(marks, clip)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    print(f"saved {len(marks)} true strikes to {clip.path('truth.csv')}")
    if det and marks:
        r = compare(det, list(marks), cfg["handcheck"]["match_tol_frames"])
        print(format_report(clip.name, r))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clip", required=True, help="Stage 2 clip folder or video file")
    run(Clip(p.parse_args().clip), load_config())


if __name__ == "__main__":
    main()
