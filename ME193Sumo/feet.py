"""
feet.py

The robot walks on crank feet instead of wheels, so "stopped" should mean
"feet flat in their base position", not wherever the motors happened to be.

Each motor's base position is an arc of its absolute encoder angle
(0..359°), found with tests/hw_feet_calibrate.py and saved to
feet_calibration.json. Arcs may wrap past 0° (e.g. 340° → 20°).

When the driver lets go of the stick, `settle_speed()` keeps each motor
turning the way it last turned until it lands inside the *middle* of its
arc (config.SETTLE_TARGET_FRACTION, e.g. the center 50%).

Pure logic: tested in tests/test_feet.py.
"""

import json
import os
from dataclasses import dataclass

CALIBRATION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feet_calibration.json")


@dataclass(frozen=True)
class Arc:
    """Angles from `start` going up (increasing degrees) to `end`, inclusive."""
    start: float
    end: float

    @property
    def width(self):
        return (self.end - self.start) % 360

    def contains(self, angle):
        return (angle - self.start) % 360 <= self.width

    def middle(self, fraction):
        """The centered sub-arc covering `fraction` of this arc's width
        (0.5 → trims 25% off each end)."""
        trim = self.width * (1 - fraction) / 2
        return Arc((self.start + trim) % 360, (self.end - trim) % 360)


def arc_from_marks(a, b, travelled):
    """The two calibration marks → the arc the foot actually swept.

    a, b:      absolute angles saved at the first and second mark
    travelled: signed degrees turned from mark a to mark b (difference of
               the motor's cumulative `position` counter). Its sign says
               which way round the circle the base range goes -- the two
               marks alone can't tell a 170° range from its 190° complement.
    """
    a, b = a % 360, b % 360
    return Arc(a, b) if travelled >= 0 else Arc(b, a)


@dataclass(frozen=True)
class FeetState:
    """One tick of foot info for the decision tree."""
    left_pos: float
    right_pos: float
    left_arc: Arc
    right_arc: Arc
    left_dir: int = 0     # sign of the last non-zero command sent to that side
    right_dir: int = 0


def settle_speed(pos, arc, last_dir, speed, allow_reverse=True):
    """Speed for one foot while the driver isn't driving.

    In the arc → 0 (stop). Otherwise keep going the way it last went
    (forward if it hasn't moved yet). If reverse isn't allowed (edge
    guard), settle forward instead -- a crank reaches its arc either way.
    """
    if arc.contains(pos):
        return 0
    direction = last_dir if last_dir != 0 else 1
    if direction < 0 and not allow_reverse:
        direction = 1
    return direction * speed


def sign(x):
    return (x > 0) - (x < 0)


def save_calibration(left_arc, right_arc, path=CALIBRATION_FILE):
    data = {side: {"start": arc.start, "end": arc.end}
            for side, arc in (("left", left_arc), ("right", right_arc))}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_calibration(path=CALIBRATION_FILE):
    """(left_arc, right_arc), or None if the feet haven't been calibrated."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return tuple(Arc(data[side]["start"], data[side]["end"]) for side in ("left", "right"))
