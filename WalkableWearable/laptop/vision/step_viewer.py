"""Stage 3: vision-only step detector -- live webcam or recorded clip.

    python -m laptop.vision.step_viewer --camera              # live, watch it count
    python -m laptop.vision.step_viewer --camera --record normal_walk
                                                       # live + save a Stage 2 clip
    python -m laptop.vision.step_viewer --clip data/clips/<folder>
    python -m laptop.vision.step_viewer --clip my_phone_video.mov --save-video
    python -m laptop.vision.step_viewer --clip <folder> --no-display   # fastest

For clips, pose landmarks are cached in landmarks_<model>.parquet: re-running after
changing a steps: threshold in config.yaml is instant (--repose to redo pose).
Step events go to steps.csv (t_ns, leg, frame_idx, confidence).

Camera side-on to the walking course, whole legs in frame, good light.
Keys: q / Esc quit, space pause (clip mode).
"""

import argparse
import time

import cv2
import pandas as pd

from laptop.config import load_config
from laptop.vision.clip import Clip
from laptop.vision.overlay import draw_counts, draw_flash, draw_legs
from laptop.vision.pose import (PoseTracker, landmarks_row, load_landmarks, row_landmarks,
                                save_landmarks)
from laptop.vision.steps import make_detector

WINDOW = "Steps (q quit, space pause)"


def save_steps(events, path):
    pd.DataFrame([{"t_ns": e.t_ns, "leg": e.leg, "frame_idx": e.frame_idx,
                   "confidence": e.confidence} for e in events],
                 columns=["t_ns", "leg", "frame_idx", "confidence"]).to_csv(path, index=False)


def detect_clip(clip, cfg, display=True, save_video=False, repose=False):
    """Run pose (or use the cache) + the step detector over every frame."""
    cache = clip.landmarks_path(cfg["pose"]["model"])
    cached = None
    if cache.exists() and not repose:
        cached = load_landmarks(cache)
        expected = len(clip.t_ns) if clip.t_ns is not None else None
        if expected is not None and len(cached) < expected:
            print(f"cached landmarks cover {len(cached)}/{expected} frames -- redoing pose")
            cached = None
        else:
            print(f"using cached landmarks {cache} ({len(cached)} frames)")
    need_frames = cached is None or display or save_video
    tracker = PoseTracker(cfg) if cached is None else None
    det = make_detector(cfg["steps"])
    rows, writer = [], None
    t0, paused, quit_early = time.monotonic(), False, False

    if need_frames:
        source = clip.frames()
    else:  # cache only: no decoding at all
        source = ((int(r.frame_idx), int(r.t_ns), None) for r in cached.itertuples())
    for idx, t_ns, frame in source:
        if cached is not None:
            if idx >= len(cached):
                break
            lm = row_landmarks(cached.iloc[idx])
        else:
            lm = tracker.process(frame, t_ns)
            rows.append(landmarks_row(idx, t_ns, lm))
        det.update(idx, t_ns, lm)

        if frame is not None and (display or save_video):
            # Events are confirmed a few frames late; flash them from their own frame.
            draw_legs(frame, lm, cfg)
            draw_flash(frame, det.events[-4:], lm, t_ns, cfg)
            elapsed = time.monotonic() - t0
            draw_counts(frame, det.counts, [f"{clip.name}  frame {idx}",
                                            f"processing {(idx + 1) / max(elapsed, 1e-3):.1f} fps"], cfg)
            if save_video:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(str(clip.path("annotated.mp4")),
                                             cv2.VideoWriter_fourcc(*"mp4v"), clip.fps, (w, h))
                writer.write(frame)
            if display:
                cv2.imshow(WINDOW, frame)
                key = cv2.waitKey(0 if paused else 1) & 0xFF
                if key in (ord("q"), 27):
                    quit_early = True
                    break
                if key == ord(" "):
                    paused = not paused
        if idx % 300 == 0 and not display:
            print(f"  frame {idx} ...")

    if tracker:
        tracker.close()
        if not quit_early:  # a partial cache would silently cut later runs short
            save_landmarks(rows, cache, cfg["pose"]["model"])
            print(f"saved landmarks to {cache}")
    if writer:
        writer.release()
        print(f"saved {clip.path('annotated.mp4')}")
    if display:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    if quit_early:
        print(f"{clip.name}: stopped at frame {idx} -- nothing saved (let it play to the end,"
              " or use --no-display)")
        return det
    save_steps(det.events, clip.path("steps.csv"))
    print(f"{clip.name}: L {det.counts['L']}  R {det.counts['R']}  total {det.total}"
          f"  -> {clip.path('steps.csv')}")
    return det


def run_live(cfg, record_label=None):
    from laptop.vision.capture import CameraRecorder, new_clip_dir
    out = new_clip_dir(cfg, record_label) if record_label else None
    rec = CameraRecorder(cfg, out, record_label or "live")
    rec.start()
    tracker = PoseTracker({**cfg, "pose": {**cfg["pose"], "model": cfg["pose"]["live_model"]}})
    det = make_detector(cfg["steps"])
    done, shown = [], -1
    fps_t = []
    print(f"live (q to quit){'; recording to ' + str(out) if out else ''}")
    try:
        while True:
            latest = rec.latest()
            if not latest or latest[0] == shown:
                if cv2.waitKey(2) & 0xFF in (ord("q"), 27):
                    break
                continue
            idx, t_ns, frame = latest
            shown = idx
            frame = frame.copy()
            lm = tracker.process(frame, t_ns)  # pose is slower than the camera: newest frame only
            for ev in det.update(idx, t_ns, lm):
                print(f"step {ev.leg}  frame {ev.frame_idx}  conf {ev.confidence:.2f}"
                      f"  (L {det.counts['L']}  R {det.counts['R']})")
            fps_t = [x for x in fps_t if x > time.monotonic() - 2] + [time.monotonic()]
            s = rec.stats()
            draw_legs(frame, lm, cfg)
            draw_flash(frame, det.events[-4:], lm, t_ns, cfg)
            draw_counts(frame, det.counts,
                        [f"pose {len(fps_t) / 2:.0f} fps   camera {s['fps_now']:.0f} fps"
                         + ("   REC" if out else "")], cfg)
            cv2.imshow(WINDOW, frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        tracker.close()
        rec.stop()
    print(f"live total: L {det.counts['L']}  R {det.counts['R']}  total {det.total}")
    if out:
        print(f"clip saved to {out}\n  next: python -m laptop.vision.step_viewer --clip {out}")


def main():
    cfg = load_config()
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--camera", action="store_true")
    src.add_argument("--clip", help="Stage 2 clip folder or any video file")
    p.add_argument("--record", metavar="LABEL", help="with --camera: also save a clip")
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--save-video", action="store_true", help="write annotated.mp4")
    p.add_argument("--repose", action="store_true", help="ignore cached landmarks")
    args = p.parse_args()
    if args.camera:
        run_live(cfg, args.record)
    else:
        detect_clip(Clip(args.clip), cfg, not args.no_display, args.save_video, args.repose)


if __name__ == "__main__":
    main()
