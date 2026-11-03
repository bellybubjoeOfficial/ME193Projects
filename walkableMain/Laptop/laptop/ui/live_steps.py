"""Stage 7: live IMU-only step detection -- holographic HUD.

    python -m laptop.ui.live_steps                          # live from the UNO Q
    python -m laptop.ui.live_steps --replay data/recordings/<folder>   # no hardware
    python -m laptop.ui.live_steps --model models/step_cnn.pt --duration 60

LIVE tab: step counter ring (flashes cyan = left / magenta = right on every
strike), L / R counts, cadence, last step, strike probability per leg
(solid = final, faint = still provisional), raw IMU, event log.
PARAMETERS tab: live-tunable values (apply instantly, SAVE writes
config.yaml) and the retrain-required values the loaded model was built with.
"""

import argparse
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from laptop.acquire.ringbuffer import RingBuffer
from laptop.clock import now_ns
from laptop.config import load_config, resolve, set_values
from laptop.train.dataset import Normalizer
from laptop.train.model import load as load_model
from laptop.train.stream import StreamingDetector
from laptop.ui import theme as T

G = 9.80665


# ---------------- data sources ----------------
class LiveSource:
    """The real UNO Q over MQTT (Stage 1 acquirer; also logs to data/sessions)."""

    def __init__(self, cfg):
        from laptop.acquire.acquirer import ImuAcquirer
        self.acq = ImuAcquirer(cfg)
        self.ring = self.acq.ring
        self.name = "LIVE · UNO Q"

    def start(self):
        self.acq.start()

    def stop(self):
        self.acq.stop()

    def stats(self):
        s = self.acq.stats()
        return {"link": s["connected"] and s["received"] > 0, "synced": s["offset_ns"] is not None,
                "rate": s["rate_hz_window"], "missing": s["missing"], "gaps": len(s["gaps"]),
                "net_ms": s["latency_ms_median"]}


class ReplaySource:
    """Plays a recorded session's IMU at real time, as if it were arriving now."""

    def __init__(self, cfg, folder, loop=False):
        from laptop.data.load import load_imu
        imu = load_imu(folder).dropna(subset=["t_laptop_ns"])
        self.seq = imu["seq"].to_numpy(np.int64)
        self.t = imu["t_laptop_ns"].astype("int64").to_numpy()
        self.x = imu[["ax", "ay", "az", "gx", "gy", "gz"]].to_numpy(np.float64)
        self.ring = RingBuffer(int(cfg["imu"]["ring_seconds"] * 150))
        self.net_ms = cfg["live"]["replay_latency_ms"]
        self.loop = loop
        self.name = f"REPLAY · {Path(folder).name}"
        self._stop = threading.Event()
        self.finished = False

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        self._stop.set()

    def _run(self):
        seq_offset = 0
        while not self._stop.is_set():
            t0_play, t0_rec = now_ns(), self.t[0]
            for a in range(0, len(self.seq), 10):  # 10-sample batches, like the UNO Q
                b = min(a + 10, len(self.seq))
                due = t0_play + (self.t[b - 1] - t0_rec) + int(self.net_ms * 1e6)
                while now_ns() < due:
                    if self._stop.wait(0.002):
                        return
                self.ring.append(self.t[a:b] - t0_rec + t0_play, self.seq[a:b] + seq_offset,
                                 self.x[a:b])
            if not self.loop:
                self.finished = True
                return
            seq_offset += int(self.seq[-1] - self.seq[0]) + 1000

    def stats(self):
        return {"link": True, "synced": True, "rate": 100.0, "missing": 0, "gaps": 0,
                "net_ms": self.net_ms}


# ---------------- widgets ----------------
class StepRing(QtWidgets.QWidget):
    """HUD ring around the step total; a step sends a colored shockwave."""

    def __init__(self):
        super().__init__()
        self.total = 0
        self.flash_s = 0.35
        self._flash = None  # (t0 monotonic, color)
        self._phase = 0.0
        self.setMinimumSize(320, 320)

    def step(self, total, leg):
        self.total = total
        self._flash = (time.monotonic(), QtGui.QColor(T.LEG_COLOR[leg]))

    def tick(self, dt):
        self._phase = (self._phase + dt * 18) % 360
        self.update()

    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        w, h = self.width(), self.height()
        c = QtCore.QPointF(w / 2, h / 2)
        r = min(w, h) / 2 - 14

        k = 0.0
        flash_c = QtGui.QColor(T.CYAN)
        if self._flash:
            k = max(0.0, 1 - (time.monotonic() - self._flash[0]) / self.flash_s)
            flash_c = self._flash[1]

        # core glow
        g = QtGui.QRadialGradient(c, r)
        g.setColorAt(0.0, T.rgba(flash_c, int(40 + 150 * k)))
        g.setColorAt(0.55, T.rgba(flash_c, int(10 + 50 * k)))
        g.setColorAt(1.0, T.rgba(T.BG, 0))
        p.setBrush(g)
        p.setPen(QtCore.Qt.NoPen)
        p.drawEllipse(c, r, r)

        # rotating tick ring
        p.save()
        p.translate(c)
        p.rotate(self._phase)
        for i in range(72):
            a = i * 5
            long = i % 6 == 0
            pen = QtGui.QPen(T.rgba(T.CYAN, 200 if long else 70), 2 if long else 1)
            p.setPen(pen)
            p.save()
            p.rotate(a)
            p.drawLine(QtCore.QPointF(r - (14 if long else 7), 0), QtCore.QPointF(r, 0))
            p.restore()
        p.restore()

        # counter-rotating dashed arc
        pen = QtGui.QPen(T.rgba(T.VIOLET, 160), 2, QtCore.Qt.DashLine)
        p.setPen(pen)
        p.setBrush(QtCore.Qt.NoBrush)
        rr = r * 0.80
        p.drawArc(QtCore.QRectF(c.x() - rr, c.y() - rr, 2 * rr, 2 * rr), int(-self._phase * 2 * 16), 250 * 16)

        # inner solid ring + shockwave
        p.setPen(QtGui.QPen(T.rgba(flash_c, int(90 + 165 * k)), 2 + 4 * k))
        ri = r * 0.66
        p.drawEllipse(c, ri, ri)
        if k > 0:
            rs = ri + (r - ri) * (1 - k) * 1.3
            p.setPen(QtGui.QPen(T.rgba(flash_c, int(255 * k)), 3))
            p.drawEllipse(c, rs, rs)

        # numbers
        p.setPen(QtGui.QColor(T.TEXT if k == 0 else flash_c.lighter(160)))
        f = QtGui.QFont(T.MONO, int(r * 0.36), QtGui.QFont.Bold)
        p.setFont(f)
        p.drawText(QtCore.QRectF(0, c.y() - r * 0.32, w, r * 0.5), QtCore.Qt.AlignCenter, f"{self.total}")
        p.setPen(QtGui.QColor(T.TEXT))
        f2 = QtGui.QFont(T.MONO, 10)
        f2.setLetterSpacing(QtGui.QFont.AbsoluteSpacing, 5)
        p.setFont(f2)
        p.drawText(QtCore.QRectF(0, c.y() + r * 0.2, w, 20), QtCore.Qt.AlignCenter, "STEPS")
        p.end()


class Tile(QtWidgets.QFrame):
    def __init__(self, title, color=T.CYAN, big=28):
        super().__init__()
        self.setObjectName("panel")
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.addWidget(T.caption(title))
        self.value = QtWidgets.QLabel("--")
        self.value.setStyleSheet(f"color: {color}; font-size: {big}px; font-weight: bold;")
        T.glow(self.value, color, 18)
        lay.addWidget(self.value)
        self.sub = QtWidgets.QLabel("")
        self.sub.setStyleSheet(f"color: {T.DIM}; font-size: 10px;")
        lay.addWidget(self.sub)

    def set(self, value, sub=None, color=None):
        self.value.setText(str(value))
        if color:
            self.value.setStyleSheet(f"color: {color}; font-size: {self.value.font().pixelSize() or 28}px;"
                                     " font-weight: bold;")
            T.glow(self.value, color, 18)
        if sub is not None:
            self.sub.setText(sub)


def chip(text, ok):
    color = T.GREEN if ok else T.RED
    return f"<span style='color:{color}'>●</span> <span style='color:{T.TEXT}'>{text}</span>"


# ---------------- main window ----------------
class LiveStepsWindow(QtWidgets.QMainWindow):
    def __init__(self, cfg, source, model_path, duration_s=None):
        super().__init__()
        self.cfg, self.source = cfg, source
        self.duration_s = duration_s
        model, ck = load_model(model_path)
        self.ck, self.model_path = ck, Path(model_path)
        norm = Normalizer(ck["norm"]["mean"], ck["norm"]["std"])
        self.det = StreamingDetector(model, norm, cfg, ck.get("rate_hz", 100.0))
        lc = cfg["live"]
        self.flash_s = lc["flash_s"]
        self.plot_window_s = lc["plot_window_s"]
        self._hist = deque()       # finalized (t_ns, pL, pR)
        self._hist_seq_end = None  # newest seq already in _hist
        self._log = deque(maxlen=7)
        self._last_update = 0.0
        self._last_tick = time.monotonic()
        self._fps = deque(maxlen=60)
        self.t_start = time.monotonic()
        self.latencies = []
        self.dist = self._load_distance(cfg)

        self.setWindowTitle("STRIDE · HOLO — live step inference")
        self.resize(1440, 900)
        root = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(root)
        outer.setContentsMargins(18, 14, 18, 14)
        outer.addLayout(self._header())
        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self._live_tab(), "LIVE")
        tabs.addTab(self._params_tab(), "PARAMETERS")
        outer.addWidget(tabs)
        self.setCentralWidget(root)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(33)

    # ---- layout ----
    def _header(self):
        row = QtWidgets.QHBoxLayout()
        col = QtWidgets.QVBoxLayout()
        title = QtWidgets.QLabel("◢ STRIDE · HOLO ◣")
        title.setObjectName("title")
        T.glow(title, T.CYAN, 26)
        col.addWidget(title)
        sub = QtWidgets.QLabel(f"SHIN IMU STEP INFERENCE  //  {self.source.name}  //  "
                               f"MODEL {self.model_path.name}")
        sub.setObjectName("subtitle")
        col.addWidget(sub)
        row.addLayout(col)
        row.addStretch()
        self.chips = QtWidgets.QLabel()
        self.chips.setTextFormat(QtCore.Qt.RichText)
        self.chips.setStyleSheet("font-size: 11px; letter-spacing: 2px;")
        row.addWidget(self.chips)
        return row

    def _live_tab(self):
        w = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(w)
        grid.setSpacing(12)

        left = T.panel()
        lv = QtWidgets.QVBoxLayout(left)
        lv.addWidget(T.caption("step counter"))
        self.ring = StepRing()
        self.ring.flash_s = self.flash_s
        lv.addWidget(self.ring, stretch=1)
        legs = QtWidgets.QHBoxLayout()
        self.tile_l = Tile("left", T.CYAN, 34)
        self.tile_r = Tile("right", T.MAGENTA, 34)
        legs.addWidget(self.tile_l)
        legs.addWidget(self.tile_r)
        lv.addLayout(legs)
        info = QtWidgets.QHBoxLayout()
        self.tile_cad = Tile("cadence", T.VIOLET, 24)
        self.tile_last = Tile("last step", T.TEXT, 24)
        info.addWidget(self.tile_cad)
        info.addWidget(self.tile_last)
        lv.addLayout(info)
        self.tile_dist = Tile("distance", T.GREEN, 30)
        lv.addWidget(self.tile_dist)
        grid.addWidget(left, 0, 0, 2, 1)

        right = T.panel()
        rv = QtWidgets.QVBoxLayout(right)
        rv.addWidget(T.caption("strike probability  ·  solid = final   faint = provisional"))
        pg.setConfigOptions(antialias=True)
        glw = pg.GraphicsLayoutWidget()
        glw.setBackground(T.rgba(T.PANEL, 0))
        self.p_prob = glw.addPlot(row=0, col=0)
        T.style_plot(self.p_prob, "P(strike)")
        self.p_prob.setYRange(0, 1.05, padding=0)
        self.p_prob.setXRange(-self.plot_window_s, 0, padding=0)
        self.c_prob = {leg: T.neon_curve(self.p_prob, T.LEG_COLOR[leg], 2.0, fill=True) for leg in "LR"}
        for p_ in (self.p_prob,):
            p_.setDownsampling(auto=True, mode="peak")
            p_.setClipToView(True)
        self.c_prov = {leg: self.p_prob.plot(pen=pg.mkPen(T.rgba(T.LEG_COLOR[leg], 90), width=1.5,
                                                            style=QtCore.Qt.DotLine)) for leg in "LR"}
        self.thr_line = pg.InfiniteLine(angle=0, pos=self.det.threshold,
                                        pen=pg.mkPen(T.AMBER, width=1, style=QtCore.Qt.DashLine),
                                        label="THRESHOLD", labelOpts={"color": T.AMBER, "position": 0.04})
        self.p_prob.addItem(self.thr_line)
        self.final_line = pg.InfiniteLine(angle=90, pos=-self.det.latency_s,
                                          pen=pg.mkPen(T.rgba(T.VIOLET, 160), width=1, style=QtCore.Qt.DashLine),
                                          label="FINAL ◂ ▸ LIVE", labelOpts={"color": T.VIOLET, "position": 0.93})
        self.p_prob.addItem(self.final_line)
        self.markers = pg.ScatterPlotItem(size=16, symbol="t1", pxMode=True)
        self.p_prob.addItem(self.markers)

        self.p_imu = glw.addPlot(row=1, col=0)
        T.style_plot(self.p_imu, "IMU")
        self.p_imu.setXLink(self.p_prob)
        self.p_imu.setYRange(0, 4.5, padding=0)
        self.c_acc = T.neon_curve(self.p_imu, T.GREEN, 1.3, halo=False)
        self.c_gyr = T.neon_curve(self.p_imu, T.VIOLET, 1.1, halo=False)
        self.p_imu.setDownsampling(auto=True, mode="peak")
        self.p_imu.setClipToView(True)
        legend = pg.LegendItem(offset=(10, 5), labelTextColor=T.DIM)
        legend.setParentItem(self.p_imu.getViewBox())
        legend.addItem(self.c_acc[-1], "|accel| g")
        legend.addItem(self.c_gyr[-1], "|gyro| / 200 °/s")
        glw.ci.layout.setRowStretchFactor(0, 3)
        glw.ci.layout.setRowStretchFactor(1, 2)
        rv.addWidget(glw)
        grid.addWidget(right, 0, 1)

        log = T.panel()
        lg = QtWidgets.QVBoxLayout(log)
        lg.addWidget(T.caption("event log"))
        self.log_label = QtWidgets.QLabel()
        self.log_label.setTextFormat(QtCore.Qt.RichText)
        self.log_label.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        lg.addWidget(self.log_label)
        grid.addWidget(log, 1, 1)
        grid.setColumnStretch(0, 2)
        grid.setColumnStretch(1, 5)
        grid.setRowStretch(0, 4)
        grid.setRowStretch(1, 1)
        return w

    def _params_tab(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(w)
        lay.setSpacing(14)

        live = T.panel()
        lv = QtWidgets.QVBoxLayout(live)
        lv.addWidget(T.caption("live-tunable  ·  applies instantly"))
        self.params = {}
        rows = [
            (("train", "threshold"), "THRESHOLD", "strike probability needed to count a step",
             self.det.threshold, 0.05, 0.99, 0.01),
            (("train", "refractory_s"), "REFRACTORY  s", "min time between steps on one leg",
             self.det.refractory_s, 0.10, 1.00, 0.01),
            (("live", "lookahead_s"), "LOOKAHEAD  s",
             f"wait this long after a strike before calling it  (exact = {self.det.full_lookahead / self.det.rate:.2f} s)",
             self.det.lookahead / self.det.rate, 0.05, self.det.full_lookahead / self.det.rate, 0.01),
            (("live", "bout_min_steps"), "BOUT MIN STEPS",
             "start counting after this many rhythmic steps from both legs  (1 = off)",
             float(self.det.gate.min_steps), 1.0, 10.0, 1.0),
            (("live", "bout_max_gap_s"), "BOUT MAX GAP  s", "a pause longer than this ends the walking bout",
             self.det.gate.max_gap_s, 0.5, 3.0, 0.1),
            (("live", "flash_s"), "FLASH  s", "step flash length", self.flash_s, 0.05, 1.50, 0.05),
            (("live", "plot_window_s"), "PLOT WINDOW  s", "seconds of history on screen",
             self.plot_window_s, 3.0, 30.0, 1.0),
        ]
        for key, name, desc, val, lo, hi, step in rows:
            lv.addLayout(self._param_row(key, name, desc, val, lo, hi, step))
        btns = QtWidgets.QHBoxLayout()
        save = QtWidgets.QPushButton("SAVE TO CONFIG")
        save.clicked.connect(self._save_params)
        reset = QtWidgets.QPushButton("RESET COUNTERS")
        reset.clicked.connect(self._reset_counts)
        btns.addWidget(save)
        btns.addWidget(reset)
        btns.addStretch()
        lv.addLayout(btns)
        self.save_note = QtWidgets.QLabel("")
        self.save_note.setStyleSheet(f"color: {T.GREEN};")
        lv.addWidget(self.save_note)
        lv.addStretch()
        lay.addWidget(live, stretch=3)

        fixed = T.panel()
        fv = QtWidgets.QVBoxLayout(fixed)
        fv.addWidget(T.caption("retrain-required  ·  baked into the model"))
        tc = self.ck.get("train_cfg", {})
        rf = self.det.model.receptive_field
        items = [
            ("MODEL", self.model_path.name),
            ("TRAINED ON", f"{len(self.ck.get('sessions', []))} session(s)"),
            ("SAMPLE RATE", f"{self.det.rate:.1f} Hz"),
            ("RECEPTIVE FIELD", f"{rf} samples = {rf / self.det.rate * 1000:.0f} ms"),
            ("REPORT DELAY", self._delay_text()),
            ("WINDOW", f"{tc.get('window_s')} s"),
            ("LABEL WIDTH σ", f"{tc.get('label_sigma_s')} s"),
            ("CNN WIDTH", f"{tc.get('channels')} channels"),
            ("POS WEIGHT", f"{tc.get('pos_weight')}"),
            ("ITERATIONS", f"{tc.get('iterations')}"),
            ("AUGMENT", f"gain {tc.get('augment_gain')}  noise {tc.get('augment_noise')}"),
        ]
        form = QtWidgets.QFormLayout()
        form.setVerticalSpacing(10)
        for k, v in items:
            lab = QtWidgets.QLabel(v)
            if k == "REPORT DELAY":
                self._fixed_delay = lab
            lab.setStyleSheet(f"color: {T.TEXT};")
            name = QtWidgets.QLabel(k)
            name.setStyleSheet(f"color: {T.DIM}; letter-spacing: 2px;")
            form.addRow(name, lab)
        fv.addLayout(form)
        note = QtWidgets.QLabel("Change these under train: in config.yaml, then\n"
                                "python -m laptop.train.train")
        note.setStyleSheet(f"color: {T.AMBER}; margin-top: 14px;")
        fv.addWidget(note)
        fv.addStretch()
        lay.addWidget(fixed, stretch=2)
        return w

    def _delay_text(self):
        return (f"{self.det.latency_s * 1000:.0f} ms lookahead (exact "
                f"{self.det.full_lookahead / self.det.rate * 1000:.0f}) + batching + network")

    def _param_row(self, key, name, desc, val, lo, hi, step):
        box = QtWidgets.QVBoxLayout()
        top = QtWidgets.QHBoxLayout()
        lab = QtWidgets.QLabel(name)
        lab.setStyleSheet(f"color: {T.CYAN}; letter-spacing: 3px;")
        top.addWidget(lab)
        top.addStretch()
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setSingleStep(step)
        spin.setDecimals(2)
        spin.setValue(val)
        top.addWidget(spin)
        box.addLayout(top)
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        n = int(round((hi - lo) / step))
        slider.setRange(0, n)
        slider.setValue(int(round((val - lo) / step)))
        box.addWidget(slider)
        d = QtWidgets.QLabel(desc)
        d.setStyleSheet(f"color: {T.DIM}; font-size: 10px; margin-bottom: 10px;")
        box.addWidget(d)
        slider.valueChanged.connect(lambda i: spin.setValue(lo + i * step))
        spin.valueChanged.connect(lambda v: (slider.blockSignals(True),
                                             slider.setValue(int(round((v - lo) / step))),
                                             slider.blockSignals(False), self._apply(key, v)))
        self.params[key] = spin
        return box

    def _apply(self, key, v):
        if key == ("train", "threshold"):
            self.det.threshold = v
            self.thr_line.setPos(v)
        elif key == ("train", "refractory_s"):
            self.det.refractory_s = v
        elif key == ("live", "lookahead_s"):
            self.det.set_lookahead(v)
            self._fixed_delay.setText(self._delay_text())
        elif key == ("live", "bout_min_steps"):
            self.det.gate.min_steps = int(round(v))
        elif key == ("live", "bout_max_gap_s"):
            self.det.gate.max_gap_s = v
        elif key == ("live", "flash_s"):
            self.flash_s = v
            self.ring.flash_s = v
        elif key == ("live", "plot_window_s"):
            self.plot_window_s = v
            self.p_prob.setXRange(-v, 0, padding=0)
        self.save_note.setText("")

    def _save_params(self):
        set_values({k: (int(s.value()) if k[1] == "bout_min_steps" else round(s.value(), 3))
                    for k, s in self.params.items()})
        self.save_note.setText("saved to config.yaml ✓")

    def _load_distance(self, cfg):
        """Stage 8 step-length model (models/distance.json), or None if not built yet."""
        import json
        from laptop.train.distance import DistanceTracker
        path = resolve(cfg["distance"]["params_path"])
        if not path.exists():
            return None
        d = json.load(open(path))
        return DistanceTracker(d["method"], d["params"][d["method"]], cfg, d["err_frac"],
                               np.array(d["axis"]), cfg.get("imu_leg", "R"))

    def _reset_counts(self):
        last = self.det._scan_from
        self.det.reset()
        self.det._scan_from = last  # don't recount steps already on screen
        self.ring.total = 0
        if self.dist:
            self.dist.reset()

    # ---- loop ----
    def _tick(self):
        now_m = time.monotonic()
        dt, self._last_tick = now_m - self._last_tick, now_m
        self._fps.append(now_m)
        now = now_ns()

        if now_m - self._last_update >= 1 / self.cfg["live"]["update_hz"]:
            self._last_update = now_m
            t, seq, data = self.source.ring.snapshot(since_ns=now - int((self.cfg["live"]["buffer_s"] + 1) * 1e9))
            for ev in self.det.update(t, seq, data, now_ns=now):
                self._on_step(ev)
            self._grow_history()

        self.ring.tick(dt)
        self._draw_plots(now)
        self._draw_status(now)
        if self.duration_s and now_m - self.t_start >= self.duration_s:
            self.close()

    def _on_step(self, ev):
        self.latencies.append(ev.latency_ms)
        if self.dist:
            gyro = None
            if self.dist.method in ("pendulum", "swing_speed") and self.dist._last_mine is not None:
                t, _, data = self.source.ring.snapshot(since_ns=self.dist._last_mine)
                gyro = data[t < ev.t_ns, 3:]
            self.dist.add(ev.t_ns, ev.leg, gyro, self.det.rate)
        self.ring.step(self.det.total, ev.leg)
        c = T.LEG_COLOR[ev.leg]
        self.tile_last.set(ev.leg, f"p {ev.prob:.2f}  ·  {ev.latency_ms:.0f} ms after strike", c)
        stamp = time.strftime("%H:%M:%S")
        self._log.appendleft(f"<span style='color:{c}'>▲ STEP {ev.leg}</span>"
                             f"<span style='color:{T.DIM}'>  {stamp}  ·  p {ev.prob:.2f}  ·  "
                             f"Δ {ev.latency_ms:.0f} ms  ·  #{self.det.total}</span>")
        self.log_label.setText("<br>".join(self._log))

    def _grow_history(self):
        if self.det.view is None:
            return
        t_grid, probs, final_end = self.det.view
        if final_end < 0:
            return
        # Append newly finalized samples (by time; t_grid is monotonic).
        start_t = self._hist[-1][0] if self._hist else -1
        for i in range(final_end + 1):
            if t_grid[i] > start_t:
                self._hist.append((int(t_grid[i]), float(probs[i, 0]), float(probs[i, 1])))
        cutoff = now_ns() - int(31 * 1e9)
        while self._hist and self._hist[0][0] < cutoff:
            self._hist.popleft()

    def _draw_plots(self, now):
        if self._hist:
            h = np.array(self._hist)
            x = (h[:, 0] - now) / 1e9
            for k, leg in enumerate("LR"):
                for item in self.c_prob[leg]:
                    item.setData(x, h[:, k + 1])
        if self.det.view is not None:
            t_grid, probs, final_end = self.det.view
            a = max(0, final_end)
            x = (t_grid[a:] - now) / 1e9
            for k, leg in enumerate("LR"):
                self.c_prov[leg].setData(x, probs[a:, k])
            if final_end >= 0:
                self.final_line.setPos((t_grid[final_end] - now) / 1e9)
        spots = [{"pos": ((e.t_ns - now) / 1e9, min(e.prob + 0.05, 1.04)),
                  "brush": pg.mkBrush(T.LEG_COLOR[e.leg]), "pen": pg.mkPen(T.rgba("#ffffff", 120))}
                 for e in self.det.events[-40:] if (now - e.t_ns) / 1e9 < self.plot_window_s]
        self.markers.setData(spots)

        t, _, data = self.source.ring.snapshot(since_ns=now - int(self.plot_window_s * 1e9))
        if len(t):
            x = (t - now) / 1e9
            for item in self.c_acc:
                item.setData(x, np.linalg.norm(data[:, :3], axis=1) / G)
            for item in self.c_gyr:
                item.setData(x, np.linalg.norm(data[:, 3:], axis=1) / 200)

    def _draw_status(self, now):
        s = self.source.stats()
        fps = (len(self._fps) - 1) / max(self._fps[-1] - self._fps[0], 1e-3) if len(self._fps) > 2 else 0
        med_lat = np.median(self.latencies[-20:]) if self.latencies else None
        self.chips.setText("   ".join([
            chip("LINK", s["link"]), chip("SYNC", s["synced"]),
            chip(f"IMU {s['rate'] or 0:.0f} Hz", (s["rate"] or 0) > 90),
            chip(f"LOSS {s['missing']}", s["missing"] == 0),
            chip(f"DELAY {med_lat:.0f} ms" if med_lat is not None else "DELAY --", med_lat is None or med_lat < 800),
            chip("WALKING" if self.det.gate.walking else f"IDLE · {len(self.det.gate.pending)} pending",
                 self.det.gate.walking),
            chip(f"UI {fps:.0f} fps", fps > 20),
        ] + ([chip("REPLAY ENDED · use --loop", False)] if getattr(self.source, "finished", False) else [])))
        self.tile_l.set(self.det.counts["L"])
        self.tile_r.set(self.det.counts["R"])
        cad = self.det.cadence_spm(now - int(self.det.latency_s * 1e9))
        self.tile_cad.set(f"{cad:.0f}" if cad else "--", "steps / min")
        if self.dist:
            per = f"{self.dist.params['L']:.2f} m/step" if self.dist.method == "fixed" else self.dist.method
            self.tile_dist.set(f"{self.dist.total_m:.1f} m  ± {self.dist.band_m:.1f}",
                               f"{per}  ·  ±{self.dist.err_frac * 100:.0f}% (held-out)")
        else:
            self.tile_dist.set("--", "build it: python -m laptop.train.distance_eval")
        self.ring.total = self.det.total

    def closeEvent(self, event):
        self.timer.stop()
        self.source.stop()
        self.save_events()
        super().closeEvent(event)

    def save_events(self):
        """Every counted step -> data/live/<time>_steps.csv (laptop clock, comparable to
        a Stage 4 session recorded at the same time)."""
        if not self.det.events:
            return
        import pandas as pd
        out = resolve("data/live")
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{time.strftime('%Y%m%d-%H%M%S')}_steps.csv"
        pd.DataFrame([vars(e) for e in self.det.events]).assign(source=self.source.name).to_csv(path, index=False)
        self.events_path = path
        print(f"saved {len(self.det.events)} steps to {path}")


def main():
    cfg = load_config()
    p = argparse.ArgumentParser()
    p.add_argument("--replay", help="session folder to replay instead of the live UNO Q")
    p.add_argument("--loop", action="store_true", help="replay forever")
    p.add_argument("--model", default=str(resolve(cfg["train"]["model_path"])))
    p.add_argument("--duration", type=float, default=None)
    args = p.parse_args()
    if not Path(args.model).exists():
        raise SystemExit(f"no model at {args.model} -- train one: python -m laptop.train.train")

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(T.STYLESHEET)
    source = ReplaySource(cfg, args.replay, args.loop) if args.replay else LiveSource(cfg)
    source.start()
    win = LiveStepsWindow(cfg, source, args.model, args.duration)
    win.show()
    app.exec()
    print(f"steps: L {win.det.counts['L']}  R {win.det.counts['R']}  total {win.det.total}"
          + (f"  |  median report delay {np.median(win.latencies):.0f} ms" if win.latencies else ""))


if __name__ == "__main__":
    main()
