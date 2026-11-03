"""
Stage 7 test -- live (streaming) step detection.

  --replay SESSION   stream a recorded session through the live detector (no
                     hardware, no UI) and score it against that session's labels,
                     next to the offline model on the same data. Also reports the
                     report delay (strike -> step on screen).
  --live-log CSV --session SESSION
                     score steps saved by the live window (data/live/*_steps.csv)
                     during a Stage 4 recording against that recording's labels.
  --ui SESSION       open the HUD on a replay for --seconds and report UI fps.

Usage (from ME193WalkWearable/):
    python tests/hw_stage7_live.py --replay data/recordings/<folder>
    python tests/hw_stage7_live.py --live-log data/live/<time>_steps.csv --session data/recordings/<folder>
    python tests/hw_stage7_live.py --ui data/recordings/<folder> --seconds 30

Pass criteria:
  replay   live F1 within 0.02 of offline F1, median delay <= 0.5 s
  ui       median UI refresh >= 24 fps
NOTE: until there are held-out sessions, models/step_cnn.pt was trained on
every session, so replay scores are in-sample (optimistic).
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from laptop.config import load_config, resolve
from laptop.train.dataset import Normalizer, load_session
from laptop.train.events import score
from laptop.train.model import load
from laptop.train.stream import StreamingDetector, stream_arrays
from laptop.train.train import evaluate, fmt


def check(results, ok, label, detail):
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label:<26} {detail}")


def replay(folder, cfg):
    model, ck = load(resolve(cfg["train"]["model_path"]))
    norm = Normalizer(ck["norm"]["mean"], ck["norm"]["std"])
    s = load_session(folder, cfg, verbose=False)
    in_sample = s.name in ck.get("sessions", [])
    off, _, _, _ = evaluate(model, norm, s, cfg)
    det = StreamingDetector(model, norm, cfg, s.rate_hz)
    seq = np.arange(len(s.t_ns))[s.valid]
    live = stream_arrays(det, s.t_ns[s.valid], seq, s.x[s.valid], latency_ms=cfg["live"]["replay_latency_ms"])
    r = score([(e.seq, e.leg) for e in live], s.events, s.labeled, s.rate_hz, cfg["train"]["match_tol_s"])
    delay = np.median([e.latency_ms for e in live]) if live else float("nan")
    print(f"session {s.name}  ({'IN-SAMPLE: model trained on it' if in_sample else 'held out'})")
    print(f"  offline  {fmt(off)}")
    print(f"  live     {fmt(r)}")
    print(f"  lookahead {det.lookahead / det.rate:.2f} s, median delay {delay:.0f} ms "
          f"(strike -> step reported, incl. {cfg['live']['replay_latency_ms']} ms network)")
    results = []
    check(results, r["f1"] >= off["f1"] - 0.02, "live F1 ~ offline F1", f"{r['f1']:.3f} vs {off['f1']:.3f}")
    check(results, delay <= 500, "median delay <= 0.5 s", f"{delay:.0f} ms")
    return all(results)


def live_log(csv, folder, cfg):
    ev = pd.read_csv(csv)
    s = load_session(folder, cfg, verbose=False)
    idx = np.searchsorted(s.t_ns, ev["t_ns"].to_numpy())
    pred = [(int(i), leg) for i, leg in zip(np.clip(idx, 0, len(s.t_ns) - 1), ev["leg"])]
    r = score(pred, s.events, s.labeled, s.rate_hz, cfg["train"]["match_tol_s"])
    print(f"live steps {csv} vs labels of {s.name}:\n  {fmt(r)}\n  median delay {ev['latency_ms'].median():.0f} ms")
    return True


def ui(folder, seconds, cfg):
    from PySide6 import QtCore, QtWidgets
    from laptop.ui import theme as T
    from laptop.ui.live_steps import LiveStepsWindow, ReplaySource
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(T.STYLESHEET)
    src = ReplaySource(cfg, folder)
    src.start()
    win = LiveStepsWindow(cfg, src, str(resolve(cfg["train"]["model_path"])), duration_s=seconds)
    win.show()
    fps = []
    t = QtCore.QTimer()
    t.timeout.connect(lambda: fps.append((len(win._fps) - 1) / max(win._fps[-1] - win._fps[0], 1e-3))
                      if len(win._fps) > 30 else None)
    t.start(1000)
    app.exec()
    med = float(np.median(fps)) if fps else 0.0
    results = []
    print(f"UI on replay of {folder}: {win.det.total} steps")
    check(results, med >= 24, "UI refresh >= 24 fps", f"median {med:.0f} fps")
    return all(results)


def main():
    cfg = load_config()
    p = argparse.ArgumentParser()
    p.add_argument("--replay")
    p.add_argument("--live-log")
    p.add_argument("--session")
    p.add_argument("--ui")
    p.add_argument("--seconds", type=float, default=30)
    a = p.parse_args()
    if a.replay:
        ok = replay(a.replay, cfg)
    elif a.live_log:
        ok = live_log(a.live_log, a.session, cfg)
    elif a.ui:
        ok = ui(a.ui, a.seconds, cfg)
    else:
        p.error("give --replay, --live-log or --ui")
    print(f"\nSTAGE 7 {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
