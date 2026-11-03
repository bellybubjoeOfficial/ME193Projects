"""Heel-strike detection from the heel's image trajectory (camera side-on).

Two signals (config steps.method), both in LEG LENGTHS (hip-to-ankle
distance) so thresholds don't depend on distance from the camera:

  heel_x  (default) the heel's position along the walking direction. During
          swing it slides forward fast; at heel strike it STOPS and stays put
          through stance. Works at any camera height, including a floor-level
          camera where the foot barely rises on screen. Walking direction
          comes from the hips; when the hips aren't moving (standing, turning
          around) nothing is detected.
  heel_y  the heel's height on screen (y down). During swing it comes down;
          at strike it stops on the floor. Needs the camera high enough that
          a swinging foot visibly rises.

Either way the strike is the moment a signal that was rising fast levels
off. Per leg, every frame (StrikeDetector):
  1. Smooth: centered quadratic fit over `smooth_frames` frames using the
     real timestamps -> smoothed value s and rate v (no lag; decisions are
     made smooth_frames // 2 frames late).
  2. Track s_min, the lowest value since the last strike (start of swing).
  3. Arm when the signal rises fast (v > v_arm).
  4. Strike when, armed, v drops below v_stop after rising at least
     min_travel above s_min.
  5. Refine the frame with a least-squares "hinge" fit on the raw values
     around it: a rising line meeting a flat plateau. The corner is the
     moment the heel stopped (noise averages out across the fit).
  6. Refractory: no second strike on the same leg within refractory_s.
Frames whose landmark visibility is below visibility_min are ignored; a gap
longer than max_gap_frames resets the leg so a reappearing foot can't fake
a strike.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np

LEGS = ("L", "R")
SIDE = {"L": "left", "R": "right"}  # MediaPipe names are the subject's own left/right


def params(steps_cfg, method=None):
    """Flatten config steps: shared values + the chosen method's thresholds."""
    method = method or steps_cfg["method"]
    p = {k: v for k, v in steps_cfg.items() if not isinstance(v, dict)}
    p.update(steps_cfg.get(method, {}))
    p["method"] = method
    return p


@dataclass
class StepEvent:
    t_ns: int
    leg: str
    frame_idx: int
    confidence: float


class StrikeDetector:
    """One leg: detects a fast-rising signal leveling off."""

    def __init__(self, leg, p):
        self.leg = leg
        self.p = p
        self.W = int(p["smooth_frames"]) | 1  # force odd
        self.half = self.W // 2
        self.reset()
        self.last_strike_t = None

    def reset(self):
        self._buf = deque(maxlen=self.W)        # (frame_idx, t_ns, s or nan, vis)
        self._raw = deque(maxlen=self.W * 3)    # recent visible frames (frame, t, s)
        self.armed = False
        self.s_min = np.inf
        self.gap = 0
        self.s = self.v = None

    def update(self, frame_idx, t_ns, s, vis):
        p = self.p
        valid = s is not None and np.isfinite(s) and vis is not None and vis >= p["visibility_min"]
        if valid:
            self.gap = 0
        else:
            self.gap += 1
            if self.gap > p["max_gap_frames"]:
                last = self.last_strike_t
                self.reset()
                self.last_strike_t = last
        self._buf.append((frame_idx, t_ns, s if valid else np.nan, vis if valid else 0.0))
        if valid:
            self._raw.append((frame_idx, t_ns, s))
        if len(self._buf) < self.W:
            return None

        frames, ts, ss, vs = (np.array(c) for c in zip(*self._buf))
        ok = np.isfinite(ss)
        c = self.half
        if not ok[c] or ok.sum() < p["min_valid_in_window"]:
            return None
        x = (ts[ok] - ts[c]) / 1e9
        coef = np.polyfit(x, ss[ok], 2)
        s_now, v = coef[2], coef[1]
        self.s, self.v = s_now, v
        frame_c, t_c = int(frames[c]), int(ts[c])

        self.s_min = min(self.s_min, s_now)
        if v > p["v_arm"]:
            self.armed = True
        travel = s_now - self.s_min
        if not (self.armed and v <= p["v_stop"] and travel >= p["min_travel"]):
            return None

        strike_f, strike_t = self._corner(frame_c, t_c, self.s_min + travel / 2)

        self.armed = False
        self.s_min = s_now
        if self.last_strike_t is not None and strike_t - self.last_strike_t < p["refractory_s"] * 1e9:
            return None
        self.last_strike_t = strike_t
        conf = float(np.mean(vs[ok])) * min(1.0, travel / (3 * p["min_travel"]))
        return StepEvent(strike_t, self.leg, strike_f, round(conf, 3))

    def _corner(self, frame_c, t_c, s_mid):
        """Corner of the best hinge fit  s = a + b * min(t - t_k, 0).

        Fit only the second half of the rise plus the plateau: the curved
        start of the swing would otherwise drag the corner late.
        """
        raw = list(self._raw)
        below = [i for i, r in enumerate(raw) if r[2] < s_mid]
        raw = raw[below[-1]:] if below else raw
        f, t, s = (np.array(c, dtype=np.float64) for c in zip(*raw))
        if len(f) < 6:
            return frame_c, t_c
        t = (t - t[-1]) / 1e9
        best = (np.inf, frame_c, t_c)
        for k in range(2, len(f) - 2):  # need points on both sides of the corner
            X = np.column_stack([np.ones_like(t), np.minimum(t - t[k], 0)])
            coef, *_ = np.linalg.lstsq(X, s, rcond=None)
            if coef[1] <= 0:  # must be RISING into the corner
                continue
            sse = float(np.sum((X @ coef - s) ** 2))
            if sse < best[0]:
                best = (sse, int(f[k]), int(raw[k][1]))
        return best[1], best[2]


HeelStrikeDetector = StrikeDetector  # older name


class LegSignals:
    """Landmarks (pixels) -> per-leg signal in leg lengths (see module doc)."""

    def __init__(self, p):
        self.p = p
        self._len = {leg: deque(maxlen=p["leg_len_window"]) for leg in LEGS}
        self._hip = deque(maxlen=p["hip_window"])  # (t_ns, hip-centre x px)
        self.direction = 0      # +1 walking to image right, -1 left, 0 not walking
        self.turned = False     # direction flipped on this frame

    def leg_length(self, leg):
        own = self._len[leg]
        if own:
            return float(np.median(own))
        other = self._len["R" if leg == "L" else "L"]
        return float(np.median(other)) if other else None

    def _walking_direction(self, t_ns, lm):
        vmin = self.p["visibility_min"]
        hips = [lm[k][0] for k in ("left_hip", "right_hip") if k in lm and lm[k][2] >= vmin]
        if hips:
            self._hip.append((t_ns, float(np.mean(hips))))
        L = self.leg_length("L") or self.leg_length("R")
        if len(self._hip) < max(3, self._hip.maxlen // 2) or not L:
            return 0
        t, x = (np.array(c, dtype=np.float64) for c in zip(*self._hip))
        if (t[-1] - t[0]) / 1e9 > 2 * self._hip.maxlen / 30:  # stale (person was lost)
            self._hip.clear()
            return 0
        v = np.polyfit((t - t[-1]) / 1e9, x, 1)[0] / L  # leg lengths per second
        return 0 if abs(v) < self.p["v_walk_min"] else int(np.sign(v))

    def update(self, lm, t_ns=0):
        """lm: {name: (x_px, y_px, vis)} (MediaPipe names). Returns {leg: (s, vis)}."""
        vmin = self.p["visibility_min"]
        for leg in LEGS:
            side = SIDE[leg]
            hip, ankle = lm.get(f"{side}_hip"), lm.get(f"{side}_ankle")
            if hip and ankle and min(hip[2], ankle[2]) >= vmin:
                self._len[leg].append(np.hypot(hip[0] - ankle[0], hip[1] - ankle[1]))

        direction = self._walking_direction(t_ns, lm) if self.p["method"] == "heel_x" else 1
        self.turned = direction != 0 and self.direction != 0 and direction != self.direction
        self.direction = direction

        out = {}
        for leg in LEGS:
            side = SIDE[leg]
            L = self.leg_length(leg)
            heel, toe = lm.get(f"{side}_heel"), lm.get(f"{side}_foot_index")
            if L is None or L < 1 or heel is None or direction == 0:
                out[leg] = (None, 0.0)
            elif self.p["method"] == "heel_x":
                out[leg] = (direction * heel[0] / L, heel[2])
            else:
                y, vis = heel[1], heel[2]
                if self.p["signal"] == "foot_lowest" and toe and toe[2] >= vmin and toe[1] > y:
                    y, vis = toe[1], min(vis, toe[2])
                out[leg] = (y / L, vis)
        return out


class _Spot:
    """A place where some heel has been sitting still (a candidate footprint)."""

    def __init__(self, frame_idx, t_ns, x, y, label, vis):
        self.first = (frame_idx, t_ns)
        self.last_frame = frame_idx
        self.frames, self.times = [frame_idx], [t_ns]
        self.xs, self.ys = [x], [y]
        self.labels = {"L": 0, "R": 0}
        self.labels[label] += 1
        self.vis = [vis]
        self.emitted = False

    @property
    def pos(self):
        return float(np.median(self.xs)), float(np.median(self.ys))

    def add(self, frame_idx, t_ns, x, y, label, vis):
        self.last_frame = frame_idx
        self.frames.append(frame_idx)
        self.times.append(t_ns)
        self.xs.append(x)
        self.ys.append(y)
        self.labels[label] += 1
        self.vis.append(vis)


class FootprintDetector:
    """Heel strike = a heel arriving at a NEW spot on the floor and staying there.

    Leg-label agnostic: MediaPipe often swaps left/right when the legs cross
    in a side view, which garbles any per-leg trajectory. Here both heels are
    just points; a strike is the first frame of a run of >= min_stance_frames
    where a heel sits within spot_tol (leg lengths) of one spot, at least
    min_step away from every footprint of the last footprint_memory_s. The
    leg is the majority MediaPipe label of that run; if the label is
    unclear, legs alternate. Reported min_stance_frames late.
    """

    def __init__(self, p):
        self.p = p
        self.signals = LegSignals(p)  # only for leg length
        self.spots = []        # live candidates
        self.footprints = []   # (t_ns, x, y) of emitted footprints
        self._recent = deque(maxlen=2 * p["min_stance_frames"] + 2)  # (frame, [(x, y)]) heel points
        self._seen_from = None  # first frame heels were visible after the last gap
        self.counts = {leg: 0 for leg in LEGS}
        self.events = []
        self.direction = 0
        self.turned = False

    def _leg_length(self, lm):
        self.signals.update(lm)
        L = self.signals.leg_length("L") or self.signals.leg_length("R")
        if L is None:  # hips out of frame: shank is ~half a leg
            vmin = self.p["visibility_min"]
            shanks = [np.hypot(lm[f"{s}_knee"][0] - lm[f"{s}_ankle"][0],
                               lm[f"{s}_knee"][1] - lm[f"{s}_ankle"][1]) * 2
                      for s in ("left", "right")
                      if f"{s}_knee" in lm and f"{s}_ankle" in lm
                      and min(lm[f"{s}_knee"][2], lm[f"{s}_ankle"][2]) >= vmin]
            L = float(np.median(shanks)) if shanks else None
        return L

    def update(self, frame_idx, t_ns, lm):
        p = self.p
        lm = lm or {}
        L = self._leg_length(lm)
        new = []
        if self._seen_from is None:
            self._seen_from = frame_idx
        if L:
            tol, vmin = p["spot_tol"] * L, p["visibility_min"]
            for leg in LEGS:
                heel = lm.get(f"{SIDE[leg]}_heel")
                if not heel or heel[2] < vmin:
                    continue
                x, y = heel[0], heel[1]
                near = [s for s in self.spots if s.last_frame < frame_idx
                        and np.hypot(x - s.pos[0], y - s.pos[1]) <= tol]
                if near:
                    min(near, key=lambda s: np.hypot(x - s.pos[0], y - s.pos[1])).add(
                        frame_idx, t_ns, x, y, leg, heel[2])
                else:
                    spot = _Spot(frame_idx, t_ns, x, y, leg, heel[2])
                    # A foot already planted when the walker comes into view, or one
                    # that never moved, still owns its spot but is not a new strike.
                    spot.emitted = not (self._came_from_elsewhere(x, y, L)
                                        and frame_idx > self._seen_from + 2)
                    self.spots.append(spot)
            for s in self.spots:
                if not s.emitted and len(s.xs) >= p["min_stance_frames"]:
                    s.emitted = True
                    ev = self._emit(s, L)
                    if ev:
                        new.append(ev)
        vmin = p["visibility_min"]
        pts = [lm[f"{SIDE[l]}_heel"][:2] for l in LEGS
               if f"{SIDE[l]}_heel" in lm and lm[f"{SIDE[l]}_heel"][2] >= vmin]
        self._recent.append((frame_idx, pts))
        if not pts:
            self._seen_from = None
        # Drop spots no heel has visited recently.
        self.spots = [s for s in self.spots if frame_idx - s.last_frame <= p["max_gap_frames"]]
        return new

    def _came_from_elsewhere(self, x, y, L):
        """A footprint needs a heel that moved here: some heel was >= min_step
        away in the last few frames. Standing still (feet together) never is."""
        far = self.p["min_step"] * L
        return any(np.hypot(x - px, y - py) >= far for _, pts in self._recent for px, py in pts)

    def _emit(self, spot, L):
        p = self.p
        x, y = spot.pos
        # The spot opens while the heel is still slowing in; the strike is the
        # first frame the heel is within arrive_tol of where it ends up.
        d = np.hypot(np.array(spot.xs) - x, np.array(spot.ys) - y)
        k = int(np.argmax(d <= p["arrive_tol"] * L))
        frame_idx, t_ns = spot.frames[k], spot.times[k]
        memory = p["footprint_memory_s"] * 1e9
        self.footprints = [f for f in self.footprints if t_ns - f[0] <= memory]
        if any(np.hypot(x - fx, y - fy) < p["min_step"] * L for _, fx, fy in self.footprints):
            return None  # same footprint again (e.g. a heel that wobbled off and back)
        self.footprints.append((t_ns, x, y))
        n_l, n_r = spot.labels["L"], spot.labels["R"]
        leg = "L" if n_l > n_r else "R"
        share = max(n_l, n_r) / (n_l + n_r)
        if share < p["label_majority"] and self.events:
            leg = "R" if self.events[-1].leg == "L" else "L"  # unclear: alternate
        ev = StepEvent(int(t_ns), leg, int(frame_idx),
                       round(float(np.mean(spot.vis)) * share, 3))
        self.counts[leg] += 1
        self.events.append(ev)
        return ev

    @property
    def total(self):
        return sum(self.counts.values())


def make_detector(steps_cfg):
    """The detector config steps.method asks for."""
    nested = any(isinstance(v, dict) for v in steps_cfg.values())  # raw config section
    p = params(steps_cfg) if nested else steps_cfg
    return FootprintDetector(p) if p["method"] == "footprint" else StepDetector(p)


class StepDetector:
    """Both legs: feed landmarks per frame, get StepEvents back.

    Pass the whole config `steps:` section, or an already-flattened params().
    """

    def __init__(self, steps_cfg):
        nested = any(isinstance(v, dict) for v in steps_cfg.values())
        p = params(steps_cfg) if nested else steps_cfg
        self.p = p
        self.signals = LegSignals(p)
        self.legs = {leg: StrikeDetector(leg, p) for leg in LEGS}
        self.counts = {leg: 0 for leg in LEGS}
        self.events = []

    def update(self, frame_idx, t_ns, lm):
        new = []
        signals = self.signals.update(lm or {}, t_ns)
        if self.signals.turned:  # a turnaround flips the signal; start the legs fresh
            for det in self.legs.values():
                last = det.last_strike_t
                det.reset()
                det.last_strike_t = last
        for leg, (s, vis) in signals.items():
            ev = self.legs[leg].update(frame_idx, t_ns, s, vis)
            if ev:
                self.counts[leg] += 1
                self.events.append(ev)
                new.append(ev)
        return new

    @property
    def total(self):
        return sum(self.counts.values())
