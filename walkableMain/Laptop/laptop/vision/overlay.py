"""Drawing: leg skeleton, step flash with L/R label, counts."""

import cv2

from laptop.vision.steps import LEGS, SIDE

CHAIN = ["hip", "knee", "ankle", "heel", "foot_index", "ankle"]
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _text(img, text, org, scale=0.8, color=(255, 255, 255), thick=2):
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thick + 3)
    cv2.putText(img, text, org, FONT, scale, color, thick)


def leg_colors(cfg):
    o = cfg["overlay"]
    return {"L": tuple(o["left_color"]), "R": tuple(o["right_color"])}


def draw_legs(img, lm, cfg):
    if not lm:
        _text(img, "no person", (12, img.shape[0] - 20), color=(0, 0, 255))
        return
    vmin = cfg["steps"]["visibility_min"]
    colors = leg_colors(cfg)
    for leg in LEGS:
        side = SIDE[leg]
        pts = [lm.get(f"{side}_{part}") for part in CHAIN]
        for a, b in zip(pts, pts[1:]):
            if a and b:
                good = min(a[2], b[2]) >= vmin
                cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                         colors[leg] if good else (128, 128, 128), 3 if good else 1)
        for part in ("ankle", "heel", "foot_index"):
            p = lm.get(f"{side}_{part}")
            if p:
                good = p[2] >= vmin
                cv2.circle(img, (int(p[0]), int(p[1])), 6 if part == "heel" else 4,
                           colors[leg] if good else (128, 128, 128), -1 if good else 1)


def draw_flash(img, recent_events, lm, now_ns, cfg):
    """Border flash + big L/R at the foot for events within flash_s."""
    colors = leg_colors(cfg)
    flash_ns = cfg["overlay"]["flash_s"] * 1e9
    h, w = img.shape[:2]
    for ev in recent_events:
        if now_ns - ev.t_ns > flash_ns:
            continue
        c = colors[ev.leg]
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), c, 18)
        heel = lm.get(f"{SIDE[ev.leg]}_heel") if lm else None
        org = (int(heel[0]) - 25, int(heel[1]) - 40) if heel else ((w // 2 - 40) if ev.leg == "L" else (w // 2 + 40), h // 2)
        _text(img, ev.leg, org, scale=2.5, color=c, thick=6)


def draw_counts(img, counts, extra_lines=(), cfg=None):
    colors = leg_colors(cfg) if cfg else {"L": (255, 255, 255), "R": (255, 255, 255)}
    _text(img, f"L {counts['L']}", (12, 40), 1.2, colors["L"], 3)
    _text(img, f"R {counts['R']}", (130, 40), 1.2, colors["R"], 3)
    _text(img, f"total {counts['L'] + counts['R']}", (250, 40), 1.2, (255, 255, 255), 3)
    for i, line in enumerate(extra_lines):
        _text(img, line, (12, 80 + 30 * i), 0.7)
