"""Stage 4: record IMU + webcam together into one session folder.

    python -m laptop.sync.record --speed normal --course 10 --leg R
    python -m laptop.sync.record --speed slow --course 10 --leg R --notes "tape course" --minutes 2

Session folder data/recordings/<YYYYmmdd-HHMMSS>_<speed>/:
    imu.parquet sync_log.csv status.jsonl gaps.csv imu_meta.json   (Stage 1)
    video.mp4 frames.csv camera.json                               (Stage 2)
    meta.json        course length, speed, IMU leg, notes, run summary
    stomp.json stomp.png sync_correction.json                      (stomp check)

Protocol (shown on screen): stand still -> STOMP the IMU foot -> walk ->
stand still -> STOMP -> press q. Both clocks are the laptop's
time.monotonic_ns(); the stomps check that they agree to within a frame.
"""

import argparse
import json
import shutil
import time

import cv2
import numpy as np

from laptop.acquire.acquirer import ImuAcquirer
from laptop.clock import now_ns
from laptop.config import load_config, resolve
from laptop.vision.capture import CameraRecorder

G = 9.80665
WINDOW = "Synced recording (q = stop after the final stomp)"


def wait_for_imu(acq, timeout_s):
    t_end = time.monotonic() + timeout_s
    while time.monotonic() < t_end:
        s = acq.stats()
        if s["received"] > 0 and s["offset_ns"] is not None:
            return s
        print(f"  waiting: MQTT {'connected' if s['connected'] else 'not connected'}, "
              f"{s['received']} IMU samples, clock {'synced' if s['offset_ns'] is not None else 'not synced'}")
        time.sleep(1)
    return None


def draw_overlay(frame, acq, cam, cfg, elapsed_s):
    h, w = frame.shape[:2]
    s, c = acq.stats(), cam.stats()
    trace_s = cfg["record"]["imu_trace_s"]
    min_g = cfg["stomp"]["min_g"]

    # accel magnitude trace, bottom strip
    box_h = 140
    y0 = h - box_h
    cv2.rectangle(frame, (0, y0), (w, h), (20, 20, 20), -1)
    t, _, data = acq.ring.snapshot(since_ns=now_ns() - int(trace_s * 1e9))
    if len(t) > 1:
        dyn = np.linalg.norm(data[:, :3], axis=1) / G - 1
        full = max(10.0, min_g * 2)
        xs = ((t - t[-1]) / 1e9 / trace_s + 1) * (w - 20) + 10
        ys = y0 + box_h - 10 - np.clip(dyn, 0, full) / full * (box_h - 20)
        cv2.polylines(frame, [np.column_stack([xs, ys]).astype(np.int32)], False, (0, 255, 255), 2)
        thr_y = int(y0 + box_h - 10 - min_g / full * (box_h - 20))
        cv2.line(frame, (10, thr_y), (w - 10, thr_y), (0, 0, 255), 1)
        cv2.putText(frame, f"stomp >= {min_g:g} g   now {dyn[-1]:+.1f} g   max(5s) {dyn.max():.1f} g",
                    (14, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    lines = [
        f"REC {elapsed_s:5.0f} s   video {c['fps_now']:.1f} fps  dropped {c['dropped_frames']}",
        f"IMU {s['rate_hz_window'] or 0:.1f} Hz  missing {s['missing']}  gaps {len(s['gaps'])}  "
        f"rtt {s['rtt_ms'] or 0:.0f} ms  {'MQTT OK' if s['connected'] else 'MQTT DOWN'}",
        "start: stand still, STOMP the IMU foot, then walk" if elapsed_s < 8 else
        "end: stand still, STOMP, then press q",
    ]
    for i, text in enumerate(lines):
        y = 32 + 32 * i
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255) if i == 0 else (255, 255, 255), 2)
    return frame


def record(cfg, speed, course_m, leg, notes="", minutes=None, label=None):
    folder = resolve(cfg["record"]["dir"]) / f"{time.strftime('%Y%m%d-%H%M%S')}_{label or speed}"
    folder.mkdir(parents=True)
    started_wall = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    acq = ImuAcquirer(cfg, out_dir=folder)
    acq.start()
    print("connecting to the IMU ...")
    if not wait_for_imu(acq, cfg["record"]["wait_sync_s"]):
        acq.stop()
        shutil.rmtree(folder, ignore_errors=True)
        raise SystemExit("no IMU data / clock sync -- is mosquitto running and the UNO Q streaming?")

    cam = CameraRecorder(cfg, folder, label or speed, meta_name="camera.json")
    cam.start()
    t0 = time.monotonic()
    stop_reason = "duration"
    print(f"recording to {folder}")
    shown = -1
    try:
        while minutes is None or time.monotonic() - t0 < minutes * 60:
            latest = cam.latest()
            if latest and latest[0] != shown:
                shown = latest[0]
                frame = draw_overlay(latest[2].copy(), acq, cam, cfg, time.monotonic() - t0)
                cv2.imshow(WINDOW, frame)
            if cv2.waitKey(5) & 0xFF in (ord("q"), 27):
                stop_reason = "q pressed"
                break
    except KeyboardInterrupt:
        stop_reason = "Ctrl+C"
    finally:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        cam.stop()
        imu_stats = acq.stats()
        acq.stop()

    c = cam.stats()
    meta = {
        "speed": speed, "course_length_m": course_m, "imu_leg": leg, "notes": notes,
        "started_wall": started_wall, "stopped_wall": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "stop_reason": stop_reason, "duration_s": round(time.monotonic() - t0, 1),
        "device": imu_stats["device"], "device_session": imu_stats["session"],
        "device_sessions_seen": imu_stats["sessions_seen"],
        "imu": {"received": imu_stats["received"], "expected": imu_stats["expected"],
                "gaps": len(imu_stats["gaps"]), "rate_hz": imu_stats["rate_hz_run"],
                "clock_offset_ns": imu_stats["offset_ns"], "rtt_ms": imu_stats["rtt_ms"]},
        "video": {"frames": c["frames"], "written": c["written"], "fps": c["fps_run"],
                  "dropped_frames": c["dropped_frames"], "late_stamps": c["late_stamps"],
                  "size": c["size"]},
        "clock": "all *_laptop_ns / t_ns columns are laptop time.monotonic_ns()",
        "config": cfg,
    }
    with open(folder / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"saved {folder}  ({meta['duration_s']} s, {c['frames']} frames, "
          f"{imu_stats['received']} IMU samples, stopped: {stop_reason})")
    return folder


def ask(prompt, cast=str, choices=None):
    while True:
        v = input(prompt).strip()
        try:
            v = cast(v)
        except ValueError:
            continue
        if not choices or v in choices:
            return v


def main():
    cfg = load_config()
    p = argparse.ArgumentParser()
    p.add_argument("--speed", help="speed label, e.g. slow / normal / jog")
    p.add_argument("--course", type=float, help="course length in metres")
    p.add_argument("--leg", choices=["L", "R"], help="which shin the IMU is on")
    p.add_argument("--notes", default="")
    p.add_argument("--label", help="folder name suffix (default: the speed)")
    p.add_argument("--minutes", type=float, default=None, help="stop automatically")
    p.add_argument("--no-check", action="store_true", help="skip the stomp check afterwards")
    args = p.parse_args()

    speed = args.speed or ask("speed label (slow/normal/jog/...): ")
    course = args.course if args.course is not None else ask("course length (m): ", float)
    leg = args.leg or ask("IMU on which leg (L/R): ", str.upper, {"L", "R"})
    folder = record(cfg, speed, course, leg, args.notes, args.minutes, args.label)
    if not args.no_check:
        from laptop.sync.stomp import analyze, format_report
        print("\nstomp check ...")
        print(format_report(analyze(folder, cfg), cfg))
        print(f"plot: {folder / 'stomp.png'}")


if __name__ == "__main__":
    main()
