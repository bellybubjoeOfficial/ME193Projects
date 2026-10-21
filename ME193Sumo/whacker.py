"""
whacker.py

Single Motor whacker that swings left/right.

- Manual: RT swings toward the strike end, LT toward base, speed
  proportional to how far the trigger is pulled. Both pulled cancel out.
  Released → stop (hold). Once calibrated, manual moves are hub-side
  position moves to the end point (manual_move), so the arm can never go
  past base or the strike end -- same limits as the automatic strike.
- Auto: the arm rests at BASE (center + 90° to one side). When the UNO Q
  distance sensor sees something within WHACK_TRIGGER_MM, it STRIKES: one
  sweep through center to the far end (center - 90°). It goes back to base
  once the driver hasn't touched RT/LT for WHACKER_MANUAL_HOLD_S.
  Moves are hub-side position control (see ArmMotion).

- Blind zones: at some angles the whacker itself sits in front of the
  distance sensor. Those absolute-encoder arcs come from
  tests/hw_whacker_calibrate.py (whacker_calibration.json); while the
  whacker is inside one, the distance reading is ignored (treated as unknown).

Pure logic (positions and time are passed in): tested in
tests/test_whacker.py.
"""

import json
import math
import os
from dataclasses import dataclass

import config
from feet import Arc

CALIBRATION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "whacker_calibration.json")


def whacker_speed(rt, lt, max_speed=None):
    """rt, lt: 0..1 trigger values → signed motor speed."""
    max_speed = config.WHACK_SPEED if max_speed is None else max_speed
    return round((rt - lt) * max_speed)


class ProximityTrigger:
    """Fires once when the distance first drops to WHACK_TRIGGER_MM or less.
    Re-arms only after a valid reading beyond WHACK_REARM_MM, so an
    opponent sitting close (or a noisy reading near 200 mm) doesn't cause
    whack after whack. Unknown distance (None) neither fires nor re-arms."""

    def __init__(self, trigger_mm=None, rearm_mm=None):
        self.trigger_mm = config.WHACK_TRIGGER_MM if trigger_mm is None else trigger_mm
        self.rearm_mm = config.WHACK_REARM_MM if rearm_mm is None else rearm_mm
        self.armed = True

    def update(self, distance_mm):
        if distance_mm is None:
            return False
        if self.armed and distance_mm <= self.trigger_mm:
            self.armed = False
            return True
        if not self.armed and distance_mm > self.rearm_mm:
            self.armed = True
        return False


# ── Arm positions + moves ─────────────────────────────────────────────────────
#
#   strike end            center              base
#  center - STRIKE°   (in front of sensor)   center + BASE° (on WHACKER_BASE_SIDE)
#       ◄──────────── one strike sweeps all the way across ──────────── │
#
# Positions are the motor's relative counter (what motor_run_to_relative_position
# takes); `center_relative()` finds where center is on that counter from the
# absolute angle saved by the calibration sweep.

STRIKE, HOME = "strike", "home"


def wrapped_delta(a, b):
    """Signed shortest step from angle a to angle b (-180..180)."""
    return (b - a + 180) % 360 - 180


def center_relative(rel_now, abs_now, center_abs):
    """Center's value on the relative counter, given the arm's current
    relative and absolute readings. Valid while the arm is within ±180° of
    center (it only ever moves ±90°)."""
    return rel_now + wrapped_delta(abs_now, center_abs)


@dataclass(frozen=True)
class ArmGeometry:
    center: int                    # relative-counter value of center
    base_side: int = None          # +1 / -1: which side base is on
    base_deg: float = None
    strike_deg: float = None

    def __post_init__(self):
        for name, default in (("base_side", config.WHACKER_BASE_SIDE),
                              ("base_deg", config.WHACKER_BASE_DEG),
                              ("strike_deg", config.WHACKER_STRIKE_DEG)):
            if getattr(self, name) is None:
                object.__setattr__(self, name, default)

    @property
    def base(self):
        return round(self.center + self.base_side * self.base_deg)

    @property
    def strike_end(self):
        return round(self.center - self.base_side * self.strike_deg)

    def at_base(self, position, tol=None):
        tol = config.WHACKER_TOL_DEG if tol is None else tol
        return abs(position - self.base) <= tol


def manual_move(speed, geometry, step=None):
    """Driver's RT/LT speed → (target, speed) hub position move, so the arm
    can never pass its end points: RT (speed > 0) drives toward the strike
    end, LT (speed < 0) toward base, and the hub stops it exactly there.
    Speed magnitude is rounded to `step` (default WHACKER_MANUAL_SPEED_STEP)
    so small trigger wobbles don't flood BLE with new commands.
    speed == 0 → None (stop and hold where it is)."""
    if speed == 0:
        return None
    step = config.WHACKER_MANUAL_SPEED_STEP if step is None else step
    magnitude = max(step, min(100, round(abs(speed) / step) * step))
    target = geometry.strike_end if speed > 0 else geometry.base
    return target, magnitude


class ArmMotion:
    """One hub-side position move (motor_run_to_relative_position) to the
    strike end or to base. The hub decelerates into the target itself. The
    move ends when the arm is within `tol` of the target, or after
    `timeout_s` (e.g. it hit the opponent and can't get there)."""

    def __init__(self, geometry, timeout_s=None, tol=None):
        self.geometry = geometry
        self.timeout_s = config.WHACKER_MOVE_TIMEOUT_S if timeout_s is None else timeout_s
        self.tol = config.WHACKER_TOL_DEG if tol is None else tol
        self.phase = None
        self._started = 0.0

    @property
    def active(self):
        return self.phase is not None

    def target(self):
        if self.phase == STRIKE:
            return self.geometry.strike_end
        if self.phase == HOME:
            return self.geometry.base
        return None

    def start(self, phase, now):
        self.phase = phase
        self._started = now

    def cancel(self):
        self.phase = None

    def step(self, position, now):
        """Target to keep driving to this tick, or None once the move is done."""
        if self.phase is None:
            return None
        if abs(position - self.target()) <= self.tol or now - self._started >= self.timeout_s:
            self.phase = None
        return self.target()


# ── Blind zones (whacker in front of the distance sensor) ────────────────────

@dataclass(frozen=True)
class SweepSample:
    abs_deg: float               # whacker absolute encoder angle (0..359)
    mm: object                   # closest valid distance seen there, or None


def sees_something(mm, detect_mm=None):
    detect_mm = config.WHACKER_BLIND_DETECT_MM if detect_mm is None else detect_mm
    return mm is not None and mm <= detect_mm


def _run_arc(run, margin):
    """Arc covering every angle a run of samples reached, padded by margin.

    Walks the run accumulating small wrapped steps and keeps the min and max
    reached, so a run that turns around inside the zone (the sweep reversing
    at its end) still covers its full extent instead of cancelling out."""
    start = run[0].abs_deg % 360
    pos = lo = hi = 0.0
    for a, b in zip(run, run[1:]):
        pos += wrapped_delta(a.abs_deg, b.abs_deg)
        lo, hi = min(lo, pos), max(hi, pos)
    if hi - lo + 2 * margin >= 360:
        return Arc(0, 359.999)
    return Arc((start + lo - margin) % 360, (start + hi + margin) % 360)


def blind_arcs_from_sweep(samples, detect_mm=None, margin_deg=None):
    """Sweep samples (in the order they were taken) → list of blind Arcs.

    Each run of consecutive samples where the sensor saw something becomes
    one arc covering the full extent the angle reached during the run
    (tracked through small wrapped steps), so arcs crossing 0°, either
    encoder direction, and runs spanning the sweep's turnaround all work.
    Each arc is padded by margin_deg -- keep it larger than the sweep step.
    """
    margin = config.WHACKER_BLIND_MARGIN_DEG if margin_deg is None else margin_deg
    arcs, run = [], []
    for sample in list(samples) + [None]:            # None flushes the last run
        if sample is not None and sees_something(sample.mm, detect_mm):
            run.append(sample)
            continue
        if run:
            arcs.append(_run_arc(run, margin))
            run = []
    return arcs


def in_blind_zone(abs_deg, arcs):
    """True if the whacker is blocking the sensor. Unknown angle (NaN) counts
    as blocking -- never trust a reading we can't vouch for."""
    if abs_deg is None or (isinstance(abs_deg, float) and math.isnan(abs_deg)):
        return True
    return any(arc.contains(abs_deg) for arc in arcs)


def visible_distance(mm, whacker_abs_deg, arcs):
    """The distance reading the decision tree may use: None while the
    whacker is in a blind zone, otherwise the raw reading."""
    return None if in_blind_zone(whacker_abs_deg, arcs) else mm


def save_blind_calibration(arcs, samples, path=CALIBRATION_FILE, **info):
    data = {
        "blind_arcs": [{"start": a.start, "end": a.end} for a in arcs],
        "info": info,
        "samples": [{"abs_deg": x.abs_deg, "mm": x.mm} for x in samples],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_center_abs(path=CALIBRATION_FILE):
    """Absolute angle of the arm's center (where the calibration sweep
    started from), or None if not calibrated. Older files without
    `center_abs` derive it from the first sample, which was taken at
    center + sweep_from, using the sweep's own direction."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    info = data.get("info", {})
    if "center_abs" in info:
        return info["center_abs"]
    samples = data["samples"]
    step_dir = 1 if wrapped_delta(samples[0]["abs_deg"], samples[1]["abs_deg"]) >= 0 else -1
    return (samples[0]["abs_deg"] - step_dir * info["sweep_from"]) % 360


def load_blind_calibration(path=CALIBRATION_FILE):
    """List of blind Arcs, or None if the whacker hasn't been calibrated."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return [Arc(a["start"], a["end"]) for a in data["blind_arcs"]]
