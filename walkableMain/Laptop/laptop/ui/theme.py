"""Holographic HUD theme: deep-space background, neon cyan / magenta, glow."""

import pyqtgraph as pg
from PySide6 import QtGui, QtWidgets

BG = "#03060c"
PANEL = "#06111f"
CYAN = "#00f0ff"       # left leg
MAGENTA = "#ff2bd6"    # right leg
VIOLET = "#8a5cff"
AMBER = "#ffb000"
GREEN = "#39ff9f"
RED = "#ff3b5c"
TEXT = "#bfefff"
DIM = "#2f6a82"
LEG_COLOR = {"L": CYAN, "R": MAGENTA}
MONO = "Menlo"

STYLESHEET = f"""
QWidget {{ background: {BG}; color: {TEXT}; font-family: {MONO}, 'SF Mono', monospace; font-size: 12px; }}
QFrame#panel {{ background: rgba(6, 17, 31, 0.92); border: 1px solid rgba(0, 240, 255, 0.28);
               border-radius: 10px; }}
QLabel {{ background: transparent; }}
QLabel#caption {{ color: {DIM}; font-size: 10px; letter-spacing: 3px; }}
QLabel#title {{ color: {CYAN}; font-size: 20px; letter-spacing: 8px; }}
QLabel#subtitle {{ color: {DIM}; font-size: 10px; letter-spacing: 4px; }}
QTabWidget::pane {{ border: none; }}
QTabBar::tab {{ background: transparent; color: {DIM}; padding: 8px 22px; letter-spacing: 4px;
               border-bottom: 2px solid transparent; }}
QTabBar::tab:selected {{ color: {CYAN}; border-bottom: 2px solid {CYAN}; }}
QDoubleSpinBox {{ background: #061a2b; border: 1px solid rgba(0, 240, 255, 0.45); border-radius: 4px;
                  padding: 4px 8px; color: {CYAN}; min-width: 80px; }}
QSlider::groove:horizontal {{ height: 3px; background: #0b2a3d; border-radius: 1px; }}
QSlider::sub-page:horizontal {{ background: {CYAN}; border-radius: 1px; }}
QSlider::handle:horizontal {{ background: {CYAN}; width: 14px; height: 14px; margin: -6px 0;
                              border-radius: 7px; }}
QPushButton {{ background: rgba(0, 240, 255, 0.07); border: 1px solid {CYAN}; color: {CYAN};
              padding: 7px 18px; border-radius: 4px; letter-spacing: 3px; }}
QPushButton:hover {{ background: rgba(0, 240, 255, 0.18); }}
QPushButton:pressed {{ background: rgba(0, 240, 255, 0.32); }}
"""


def rgba(color, alpha):
    c = QtGui.QColor(color)
    c.setAlpha(alpha)
    return c


def glow(widget, color=CYAN, radius=22):
    effect = QtWidgets.QGraphicsDropShadowEffect()
    effect.setBlurRadius(radius)
    effect.setColor(QtGui.QColor(color))
    effect.setOffset(0, 0)
    widget.setGraphicsEffect(effect)


def panel():
    f = QtWidgets.QFrame()
    f.setObjectName("panel")
    return f


def caption(text):
    lab = QtWidgets.QLabel(text.upper())
    lab.setObjectName("caption")
    return lab


def style_plot(p, ylabel=""):
    p.getViewBox().setBackgroundColor(rgba(PANEL, 0))
    p.showGrid(x=True, y=True, alpha=0.12)
    for name in ("left", "bottom"):
        ax = p.getAxis(name)
        ax.setPen(pg.mkPen(rgba(CYAN, 90)))
        ax.setTextPen(pg.mkPen(DIM))
        ax.setStyle(tickFont=QtGui.QFont(MONO, 9))
    if ylabel:
        p.setLabel("left", ylabel, color=DIM)
    p.hideButtons()
    p.setMenuEnabled(False)
    p.setMouseEnabled(x=False, y=False)


def neon_curve(plot, color, width=2.0, fill=False, halo=True):
    """A wide faint glow curve under a thin bright core. Returns the items.

    The halo is drawn without antialiasing: it's a blur anyway, and wide
    antialiased lines were most of the frame time.
    """
    items = []
    if halo:
        items.append(plot.plot(pen=pg.mkPen(rgba(color, 50), width=width * 3.5), antialias=False))
    kw = {"fillLevel": 0, "brush": rgba(color, 28)} if fill else {}
    items.append(plot.plot(pen=pg.mkPen(color, width=width), **kw))
    return tuple(items)
