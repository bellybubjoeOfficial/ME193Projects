"""Stage 4: stomp alignment between the IMU and the video.

Protocol: stand still, STOMP the instrumented foot, walk the session, stand
still, STOMP again. Each stomp is found twice, both on the laptop clock:
  IMU   -- the accel-magnitude spike; impact time = where |a| - 1 g first
           crosses onset_frac of the spike's peak (interpolated).
  video -- the heel's impact frame: the stomping heel (whichever heel is
           highest on screen -- no reliance on MediaPipe's L/R labels) lifts,
           then the first frame it is back on the ground.
residual = t_video - t_imu at each stomp (positive = video timestamps late,
e.g. camera pipeline latency); drift = end residual - start residual.
The correction maps IMU laptop time to video time: t_video = t_imu + r(t),
r linear between the two stomps.

    python -m laptop.sync.stomp data/recordings/<folder>
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from laptop.config import load_config
from laptop.data.load import load_imu
from laptop.vision.pose import PoseTracker

G = 9.80665


# ---------------- pure detection ----------------
def imu_stomp(t_ns, acc, t_from, t_to, min_g=3.0, onset_frac=0.5):
    """Impact time (ns) of the biggest accel spike in [t_from, t_to], or None.

    acc: (n, 3) m/s^2. Returns (t_impact_ns, peak_g).
    """
    t_ns = np.asarray(t_ns, dtype=np.int64)
    dyn = np.linalg.norm(np.asarray(acc, dtype=float), axis=1) / G - 1.0
    idx = np.flatnonzero((t_ns >= t_from) & (t_ns <= t_to))
    if len(idx) < 3:
        return None
    k = idx[np.argmax(dyn[idx])]
    peak = dyn[k]
    if peak < min_g:
        return None
    thr = onset_frac * peak
    i = k
    while i > idx[0] and dyn[i - 1] >= thr:
        i -= 1
    if i == idx[0]:
        return int(t_ns[i]), float(peak)
    # interpolate the crossing between samples i-1 (below) and i (above)
    frac = (thr - dyn[i - 1]) / (dyn[i] - dyn[i - 1])
    return int(t_ns[i - 1] + frac * (t_ns[i] - t_ns[i - 1])), float(peak)


def video_stomp(frame_idx, t_ns, heel_top_y, t_imu, leg_len_px, window_s=1.0,
                min_lift=0.08, ground_tol=0.15):
    """Impact of the stomp near t_imu, or None.

    heel_top_y: per frame, the y (px, down) of the HIGHER heel on screen
    (nan when no heel is visible). The stomping foot lifts (y drops) and
    slams back. The landing level is taken from the frames right after the
    fastest frame-to-frame drop (not the whole window, which may include
    walking); the impact time is where the heel crosses within ground_tol
    of the lift above that level, interpolated BETWEEN frames (one frame is
    33 ms -- too coarse for a 1-frame check). Returns
    (frame_idx at/after impact, t_ns interpolated, lift_px).
    """
    f = np.asarray(frame_idx)
    t = np.asarray(t_ns, dtype=np.int64)
    y = np.asarray(heel_top_y, dtype=float)
    w = (t >= t_imu - window_s * 1e9) & (t <= t_imu + window_s * 1e9) & np.isfinite(y)
    if w.sum() < 5:
        return None
    fw, tw, yw = f[w], t[w], y[w]
    # The lift must come before the impact, so look for its top up to just after t_imu.
    pre = tw <= t_imu + 0.2e9
    if not pre.any():
        return None
    k = int(np.argmin(np.where(pre, yw, np.inf)))
    if k >= len(yw) - 2:
        return None
    # Fastest drop after the top = the slam; the landing level is just after it.
    d = np.diff(yw[k:])
    slam = k + 1 + int(np.argmax(d))
    land = (tw >= tw[slam]) & (tw <= tw[slam] + 0.2e9)
    ground = float(np.median(yw[land]))
    lift = ground - yw[k]
    if lift < min_lift * leg_len_px:
        return None
    level = ground - ground_tol * lift
    after = np.flatnonzero((np.arange(len(yw)) > k) & (yw >= level))
    if not len(after):
        return None
    j = after[0]
    a = j - 1  # last frame still above the level: interpolate the crossing
    frac = (level - yw[a]) / (yw[j] - yw[a]) if yw[j] != yw[a] else 1.0
    t_cross = int(tw[a] + frac * (tw[j] - tw[a]))
    return int(fw[j]), t_cross, float(lift)


def correction(stomps):
    """{"t_imu_ns": [...], "residual_ns": [...]} -> function t_imu -> t_video."""
    t = np.array(stomps["t_imu_ns"], dtype=np.float64)
    r = np.array(stomps["residual_ns"], dtype=np.float64)
    if len(t) == 1:
        return lambda x: np.asarray(x) + r[0]
    slope = (r[1] - r[0]) / (t[1] - t[0])
    return lambda x: np.asarray(x) + r[0] + slope * (np.asarray(x, dtype=np.float64) - t[0])


# ---------------- whole session ----------------
def heel_track(video_path, frames, windows, cfg):
    """Pose only on frames inside the windows; returns per-frame heel-top y and leg length."""
    want = np.zeros(len(frames), dtype=bool)
    t = frames["t_ns"].to_numpy()
    for a, b in windows:
        want |= (t >= a) & (t <= b)
    y_top = np.full(len(frames), np.nan)
    legs = []
    tracker = PoseTracker(cfg)
    cap = cv2.VideoCapture(str(video_path))
    vmin = cfg["steps"]["visibility_min"]
    for i in range(len(frames)):
        if not want[i]:
            if not cap.grab():
                break
            continue
        ok, frame = cap.read()
        if not ok:
            break
        lm = tracker.process(frame, int(t[i]))
        heels = [lm[k][1] for k in ("left_heel", "right_heel") if k in lm and lm[k][2] >= vmin]
        if heels:
            y_top[i] = min(heels)
        for s in ("left", "right"):
            h, a = lm.get(f"{s}_hip"), lm.get(f"{s}_ankle")
            if h and a and min(h[2], a[2]) >= vmin:
                legs.append(np.hypot(h[0] - a[0], h[1] - a[1]))
    cap.release()
    tracker.close()
    return y_top, (float(np.median(legs)) if legs else None)


def analyze(folder, cfg=None):
    cfg = cfg or load_config()
    st = cfg["stomp"]
    folder = Path(folder)
    imu = load_imu(folder).dropna(subset=["t_laptop_ns"])
    frames = pd.read_csv(folder / "frames.csv")
    t_imu = imu["t_laptop_ns"].astype("int64").to_numpy()
    acc = imu[["ax", "ay", "az"]].to_numpy()
    t_v = frames["t_ns"].to_numpy()
    t0, t1 = max(t_imu[0], t_v[0]), min(t_imu[-1], t_v[-1])
    search = st["search_s"] * 1e9

    out = {"stomps": {}, "fps": float((len(t_v) - 1) / ((t_v[-1] - t_v[0]) / 1e9))}
    imu_hits = {"start": imu_stomp(t_imu, acc, t0, t0 + search, st["min_g"], st["onset_frac"]),
                "end": imu_stomp(t_imu, acc, t1 - search, t1, st["min_g"], st["onset_frac"])}
    if imu_hits["start"] and imu_hits["end"] and imu_hits["start"][0] == imu_hits["end"][0]:
        imu_hits["end"] = None  # short session: both windows found the same stomp
    win = st["video_window_s"] * 1e9
    windows = [(h[0] - win, h[0] + win) for h in imu_hits.values() if h]
    y_top, leg_len = heel_track(folder / "video.mp4", frames, windows, cfg) if windows else (None, None)

    for name, hit in imu_hits.items():
        rec = {"imu": None, "video": None}
        if hit:
            rec["imu"] = {"t_ns": hit[0], "peak_g": round(hit[1], 2)}
            v = video_stomp(frames["frame_idx"].to_numpy(), t_v, y_top, hit[0], leg_len or 200,
                            st["video_window_s"], st["min_lift"], st["ground_tol"]) if leg_len else None
            if v:
                rec["video"] = {"frame_idx": v[0], "t_ns": v[1], "lift_px": round(v[2], 1)}
                rec["residual_ms"] = (v[1] - hit[0]) / 1e6
                rec["residual_frames"] = rec["residual_ms"] / (1000 / out["fps"])
        out["stomps"][name] = rec

    # A residual far beyond camera latency means the video found the wrong event
    # (e.g. a walking step instead of the stomp): keep it in the report, not the correction.
    for k in ("start", "end"):
        s = out["stomps"][k]
        if "residual_ms" in s and abs(s["residual_ms"]) > st["max_residual_ms"]:
            s["implausible"] = True
    both = [out["stomps"][k] for k in ("start", "end")
            if "residual_ms" in out["stomps"][k] and not out["stomps"][k].get("implausible")]
    if both:
        out["correction"] = {"t_imu_ns": [s["imu"]["t_ns"] for s in both],
                             "residual_ns": [int(s["residual_ms"] * 1e6) for s in both]}
        with open(folder / "sync_correction.json", "w") as f:
            json.dump(out["correction"], f, indent=2)
    if len(both) == 2:
        span_s = (both[1]["imu"]["t_ns"] - both[0]["imu"]["t_ns"]) / 1e9
        out["drift_ms"] = both[1]["residual_ms"] - both[0]["residual_ms"]
        out["drift_ppm"] = out["drift_ms"] / 1e3 / span_s * 1e6
        out["span_s"] = span_s
    with open(folder / "stomp.json", "w") as f:
        json.dump(out, f, indent=2)
    plot(folder, imu, frames, y_top, out)
    return out


def plot(folder, imu, frames, y_top, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [k for k, v in out["stomps"].items() if v["imu"]]
    if not names:
        return
    fig, axs = plt.subplots(1, len(names), figsize=(7 * len(names), 4), squeeze=False)
    t_imu = imu["t_laptop_ns"].astype("int64").to_numpy()
    dyn = np.linalg.norm(imu[["ax", "ay", "az"]].to_numpy(), axis=1) / G - 1
    t_v = frames["t_ns"].to_numpy()
    for ax, name in zip(axs[0], names):
        s = out["stomps"][name]
        c = s["imu"]["t_ns"]
        m = np.abs(t_imu - c) < 1e9
        ax.plot((t_imu[m] - c) / 1e6, dyn[m], ".-", label="IMU |a| - 1 g")
        ax.axvline(0, color="C0", ls="--", label="IMU impact")
        if y_top is not None:
            mv = np.abs(t_v - c) < 1e9
            ax2 = ax.twinx()
            ax2.plot((t_v[mv] - c) / 1e6, y_top[mv], "o-", color="C1", ms=3, label="heel y (px)")
            ax2.invert_yaxis()
            ax2.set_ylabel("higher heel y, px (up = lifted)")
        if s["video"]:
            ax.axvline(s["residual_ms"], color="C1", ls="--",
                       label=f"video impact ({s['residual_ms']:+.1f} ms)")
        ax.set_xlabel("ms from IMU impact (laptop clock)")
        ax.set_ylabel("g")
        ax.set_title(f"{name} stomp")
        ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(folder / "stomp.png", dpi=80)
    plt.close()


def format_report(out, cfg):
    crit = cfg["stage4_pass"]
    frame_ms = 1000 / out["fps"]
    lines = [f"camera {out['fps']:.2f} fps -> 1 frame = {frame_ms:.1f} ms"]
    for name in ("start", "end"):
        s = out["stomps"][name]
        if not s["imu"]:
            lines.append(f"{name:5s} stomp: NOT FOUND in IMU (need a spike >= {cfg['stomp']['min_g']} g)")
        elif not s["video"]:
            lines.append(f"{name:5s} stomp: IMU {s['imu']['peak_g']} g, heel impact NOT FOUND in video")
        else:
            lines.append(f"{name:5s} stomp: IMU peak {s['imu']['peak_g']:.1f} g, video frame "
                         f"{s['video']['frame_idx']} -> residual {s['residual_ms']:+.1f} ms "
                         f"({s['residual_frames']:+.2f} frames)"
                         + ("  IMPLAUSIBLE -- video matched the wrong event; not used" if s.get("implausible") else ""))
    if "drift_ms" in out:
        lines.append(f"drift {out['drift_ms']:+.1f} ms over {out['span_s']:.0f} s "
                     f"({out['drift_ppm']:+.0f} ppm); limit {crit['max_drift_ms']} ms")
    if "correction" in out:
        lines.append("correction saved to sync_correction.json (t_video = t_imu + residual, "
                     "linear between stomps)")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folder")
    args = p.parse_args()
    cfg = load_config()
    out = analyze(args.folder, cfg)
    print(format_report(out, cfg))
    print(f"plot: {Path(args.folder) / 'stomp.png'}")


if __name__ == "__main__":
    main()
