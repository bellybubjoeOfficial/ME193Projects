"""Stage 2: webcam capture with a laptop-clock timestamp on every frame.

A capture thread reads the camera and stamps each frame with now_ns() (the
same clock as Stage 1) the moment read() returns. A writer thread saves
video.mp4 and frames.csv (frame_idx, t_ns) row-for-row, so the CSV always
matches the video's frame count. The main thread only previews.

    python -m laptop.vision.capture                    # record until q
    python -m laptop.vision.capture --label slow_walk --minutes 1
    python -m laptop.vision.capture --no-record        # preview only
    python -m laptop.vision.capture --probe            # find the best stable mode

Clip folder: data/clips/<YYYYmmdd-HHMMSS>_<label>/ video.mp4 frames.csv meta.json
"""

import argparse
import csv
import json
import queue
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from laptop.clock import now_ns
from laptop.config import load_config, resolve
from laptop.vision.frametiming import find_drops, find_late_stamps, mean_fps


def open_camera(index, width, height, fps):
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        raise RuntimeError(
            f"can't open camera {index}. On macOS, give your terminal/VS Code camera access: "
            "System Settings > Privacy & Security > Camera")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def new_clip_dir(cfg, label):
    folder = resolve(cfg["camera"]["clips_dir"]) / f"{time.strftime('%Y%m%d-%H%M%S')}_{label}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


class CameraRecorder:
    def __init__(self, cfg=None, out_dir=None, label="clip", meta_name="meta.json"):
        self.cfg = cfg or load_config()
        c = self.cfg["camera"]
        self.nominal_fps = float(c["fps"])
        self.drop_factor = c["drop_factor"]
        self.cap = open_camera(c["index"], c["width"], c["height"], c["fps"])
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.out_dir = Path(out_dir) if out_dir else None
        self.label = label
        self.meta_name = meta_name

        self.frames = 0           # frames captured
        self.written = 0          # frames saved (== frames.csv rows)
        self.read_failures = 0
        self.max_queue = 0
        self.t_ns = []            # every frame's timestamp
        self._recent = deque(maxlen=int(self.nominal_fps * 2))
        self._latest = None       # (frame_idx, t_ns, frame)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._queue = queue.Queue()
        self._threads = []
        self.t_start_ns = None

    def start(self):
        self.t_start_ns = now_ns()
        if self.out_dir:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            with open(self.out_dir / self.meta_name, "w") as f:
                json.dump({"label": self.label, "camera_index": self.cfg["camera"]["index"],
                           "width": self.width, "height": self.height,
                           "nominal_fps": self.nominal_fps,
                           "reported_fps": self.cap.get(cv2.CAP_PROP_FPS),
                           "clock": "laptop time.monotonic_ns (same as IMU t_laptop_ns)",
                           "started_wall": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, f, indent=2)
            self._threads.append(threading.Thread(target=self._write_loop, daemon=True))
        self._threads.append(threading.Thread(target=self._capture_loop, daemon=True))
        for th in self._threads:
            th.start()

    def stop(self):
        self._stop.set()
        for th in self._threads:
            th.join(timeout=30)
        self.cap.release()

    def latest(self):
        with self._lock:
            return self._latest

    def _capture_loop(self):
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            t = now_ns()  # stamp immediately: this is the frame's time
            if not ok:
                self.read_failures += 1
                time.sleep(0.005)
                continue
            with self._lock:
                idx = self.frames
                self.frames += 1
                self.t_ns.append(t)
                self._recent.append(t)
                self._latest = (idx, t, frame)
            if self.out_dir:
                self._queue.put((idx, t, frame))
                self.max_queue = max(self.max_queue, self._queue.qsize())
        self._queue.put(None)

    def _write_loop(self):
        fourcc = cv2.VideoWriter_fourcc(*self.cfg["camera"]["fourcc"])
        writer = cv2.VideoWriter(str(self.out_dir / "video.mp4"), fourcc, self.nominal_fps,
                                 (self.width, self.height))
        with open(self.out_dir / "frames.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame_idx", "t_ns"])
            while True:
                item = self._queue.get()
                if item is None:
                    break
                idx, t, frame = item
                writer.write(frame)
                w.writerow([idx, t])
                self.written += 1
        writer.release()

    def stats(self):
        with self._lock:
            recent = list(self._recent)
            t_all = self.t_ns
            # Judge holes against the rate the camera actually runs at (it may
            # settle at 15 fps in low light; that's reported as low fps, not drops).
            ref_fps = mean_fps(t_all) if len(t_all) > 60 else self.nominal_fps
            drops = find_drops(t_all, ref_fps, self.drop_factor)
            late = find_late_stamps(t_all, ref_fps, self.drop_factor)
            return {
                "frames": self.frames,
                "written": self.written,
                "fps_now": mean_fps(recent),
                "fps_run": mean_fps(t_all),
                "drops": drops,
                "dropped_frames": sum(d.missing for d in drops),
                "late_stamps": len(late),
                "read_failures": self.read_failures,
                "queue": self._queue.qsize(),
                "max_queue": self.max_queue,
                "elapsed_s": (now_ns() - self.t_start_ns) / 1e9 if self.t_start_ns else 0.0,
                "size": (self.width, self.height),
            }


def draw_stats(frame, s, recording):
    lines = [f"{s['size'][0]}x{s['size'][1]}  {s['fps_now']:.1f} fps (run {s['fps_run']:.2f})",
             f"frames {s['frames']}  dropped {s['dropped_frames']} in {len(s['drops'])} holes",
             f"{'REC' if recording else 'preview'}  {s['elapsed_s']:.0f} s   writer queue {s['queue']}"]
    for i, text in enumerate(lines):
        y = 30 + 30 * i
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255) if recording else (0, 255, 0), 2)
    return frame


def preview_loop(rec, duration_s=None, window="Webcam (q to stop)", on_frame=None):
    """Show the newest frame until q/Esc or duration. on_frame(idx, t, frame) may draw on it."""
    shown = -1
    reason = "duration"
    try:
        while duration_s is None or rec.stats()["elapsed_s"] < duration_s:
            latest = rec.latest()
            if latest and latest[0] != shown:
                shown = latest[0]
                frame = latest[2].copy()
                if on_frame:
                    frame = on_frame(*latest[:2], frame)
                draw_stats(frame, rec.stats(), rec.out_dir is not None)
                cv2.imshow(window, frame)
            if cv2.waitKey(5) & 0xFF in (ord("q"), 27):
                reason = "q/Esc pressed"
                break
    finally:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    return reason


def probe(cfg, seconds=4.0):
    """Try each configured mode; report what the camera actually delivers."""
    c = cfg["camera"]
    best = None
    for w, h, fps in c["probe_modes"]:
        cap = open_camera(c["index"], w, h, fps)
        aw, ah = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        for _ in range(10):  # let exposure settle
            cap.read()
        ts = []
        t_end = time.monotonic() + seconds
        while time.monotonic() < t_end:
            ok, _ = cap.read()
            if ok:
                ts.append(now_ns())
        cap.release()
        dt = np.diff(ts) / 1e6
        measured = mean_fps(ts)
        jitter = float(np.std(dt)) if len(dt) else float("nan")
        print(f"asked {w}x{h}@{fps}: got {aw}x{ah} at {measured:.2f} fps "
              f"(interval sd {jitter:.1f} ms)")
        if measured >= 0.95 * fps and (best is None or (fps, w * h) > (best[2], best[0] * best[1])):
            best = (aw, ah, fps)
    print(f"\nbest stable mode: {best}  -> put it in config.yaml camera: width/height/fps"
          if best else "\nno mode reached 95% of its requested fps (more light helps)")


def main():
    cfg = load_config()
    p = argparse.ArgumentParser()
    p.add_argument("--label", default="clip")
    p.add_argument("--minutes", type=float, default=None)
    p.add_argument("--no-record", action="store_true")
    p.add_argument("--probe", action="store_true")
    args = p.parse_args()
    if args.probe:
        probe(cfg)
        return

    out = None if args.no_record else new_clip_dir(cfg, args.label)
    rec = CameraRecorder(cfg, out, args.label)
    rec.start()
    print(f"recording to {out}" if out else "preview only")
    preview_loop(rec, args.minutes * 60 if args.minutes else None)
    rec.stop()
    s = rec.stats()
    print(f"{s['frames']} frames, {s['fps_run']:.2f} fps, {s['dropped_frames']} dropped; "
          f"saved {s['written']} to {out}" if out else f"{s['frames']} frames")


if __name__ == "__main__":
    main()
