"""Stage 1 live window: 6 scrolling IMU plots (last 10 s) + stats panel.

    python -m laptop.ui.imu_live                 # run until you close it
    python -m laptop.ui.imu_live --duration 60   # close itself after 60 s

Lines break wherever samples are missing, so a gap is visible as a hole.
"""

import argparse
import sys
import time

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from laptop.acquire.acquirer import ImuAcquirer, format_stats
from laptop.acquire.messages import FIELDS
from laptop.clock import now_ns
from laptop.config import load_config

UNITS = {"a": "m/s²", "g": "deg/s"}
COLORS = ["#e6194b", "#3cb44b", "#4363d8"]


class ImuLiveWindow(QtWidgets.QMainWindow):
    def __init__(self, acq, duration_s=None, on_finish=None):
        super().__init__()
        self.acq = acq
        self.cfg = acq.cfg["ui"]
        self.duration_s = duration_s
        self.on_finish = on_finish
        self._frame_times = []
        self.fps_history = []
        self._finished = False
        self.setWindowTitle("Shin IMU — Stage 1 acquisition")
        self.resize(1300, 850)

        pg.setConfigOptions(antialias=False)
        plots = pg.GraphicsLayoutWidget()
        self.curves = []
        first = None
        for i, name in enumerate(FIELDS):
            # Left column accel, right column gyro.
            p = plots.addPlot(row=i % 3, col=i // 3)
            p.setLabel("left", name, units=None)
            p.setTitle(f"{name} [{UNITS[name[0]]}]", size="9pt")
            p.showGrid(x=True, y=True, alpha=0.3)
            p.setXRange(-self.cfg["plot_window_s"], 0, padding=0)
            p.enableAutoRange(axis="y")
            if first is None:
                first = p
            else:
                p.setXLink(first)
            self.curves.append(p.plot(pen=pg.mkPen(COLORS[i % 3], width=1.5)))

        self.stats_label = QtWidgets.QLabel()
        self.stats_label.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        self.stats_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.stats_label.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        layout.addWidget(plots, stretch=4)
        layout.addWidget(self.stats_label, stretch=1)
        self.setCentralWidget(central)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(int(1000 / self.cfg["refresh_hz"]))

    def refresh(self):
        now = now_ns()
        window_ns = int(self.cfg["plot_window_s"] * 1e9)
        t, seq, data = self.acq.ring.snapshot(since_ns=now - window_ns)
        if len(t):
            x = (t - now) / 1e9
            connect = np.append(np.diff(seq) == 1, True)  # break the line at gaps
            for i, curve in enumerate(self.curves):
                curve.setData(x, data[:, i], connect=connect)

        self._frame_times.append(time.monotonic())
        self._frame_times = [f for f in self._frame_times if f > self._frame_times[-1] - 2]
        ui_fps = (len(self._frame_times) - 1) / max(self._frame_times[-1] - self._frame_times[0], 1e-3)
        if len(self._frame_times) > 10:
            self.fps_history.append(ui_fps)

        s = self.acq.stats()
        text = format_stats(s) + f"\nUI       {ui_fps:.0f} fps   run {s['run_s']:.0f} s"
        if self.duration_s:
            text += f" / {self.duration_s:.0f} s"
        if s["log_folder"]:
            text += f"   log {s['log_folder']}"
        gaps = s["gaps"][-self.cfg["max_gaps_shown"]:]
        if gaps:
            text += "\nGAPS     " + "   ".join(
                f"{(now - g.t_detected_ns) / 1e9:.0f}s ago: {g.count} @ seq {g.first_missing}"
                + (f" ({g.recovered} late)" if g.recovered else "") for g in reversed(gaps))
        self.stats_label.setText(text)

        if self.duration_s and s["run_s"] >= self.duration_s and not self._finished:
            self._finished = True
            self.close()

    def closeEvent(self, event):
        self.timer.stop()
        if self.on_finish:
            s = self.acq.stats()
            s["ui_fps_median"] = float(np.median(self.fps_history)) if self.fps_history else None
            s["ui_fps_p5"] = float(np.percentile(self.fps_history, 5)) if self.fps_history else None
            self.on_finish(s)
        super().closeEvent(event)


def run(duration_s=None, on_finish=None, cfg=None):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    acq = ImuAcquirer(cfg or load_config())
    acq.start()
    win = ImuLiveWindow(acq, duration_s, on_finish)
    win.show()
    try:
        app.exec()
    finally:
        acq.stop()
    return acq


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=float, default=None, help="seconds, then close")
    args = p.parse_args()
    run(args.duration)


if __name__ == "__main__":
    main()
