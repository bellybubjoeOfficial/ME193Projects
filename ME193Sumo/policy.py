"""
policy.py

The wrestler's decision tree. Every control-loop tick, `decide()` walks
from the top question down and returns the first matching Action. Higher
branches win (safety first), and each stage of the build adds new
branches for its sensor -- see TEST_PLAN.md.

Tree v5 (+ foot encoders, + whacker, + UNO Q distance → strike from base):

    controller connected? ── no ──► STOP everything               "controller lost"
            │ yes
    B held (brake)?  ──────── yes ─► STOP feet (hard, no settling) "brake"
            │ no
    stick pulls back while ─ yes ─► reverse throttle zeroed        "edge: reverse blocked"
    reverse is blocked?             (turning still ok)
            │ no
    stick in deadzone?  ───── yes ─► feet in middle of ─ yes ─►    "idle"
            │                       base arcs?
            │ no                          │ no
            │                             └─► roll each foot on in its    "settling feet"
            │                                 last direction until in arc
            │                                 (forward if reverse blocked)
    DRIVE from left stick                                           "teleop drive"

  Whacker sub-tree (runs alongside the feet tree every tick). The arm rests
  at BASE (center + 90°); a STRIKE sweeps through center to the far end.

    controller connected? ── no ──► whacker 0, cancel any move          "off"
            │ yes
    RT or LT pulled?  ─────── yes ─► manual speed, cancel any move       "manual"
            │ no                    (the driver always wins)
    strike in progress? ───── yes ─► keep driving to the strike end      "strike"
            │ no
    opponent just came within ─ yes ► start a STRIKE (also from mid-     "strike start"
    WHACK_TRIGGER_MM (UNO Q)?         way back to base)
      (distance is ignored -- counts as unknown -- while the whacker
       itself is in a calibrated blind zone: whacker.visible_distance)
            │ no
    returning to base? ────── yes ─► keep driving to base                "home"
            │ no
    away from base and no RT/LT ─ yes ► go back to base                  "home start"
    for WHACKER_MANUAL_HOLD_S (2 s)?
            │ no
    hold where it is                                                    "off"

  Plus: alert is set on the tick the color sensor first lands on the tape,
  so main.py plays a sound and rumbles the controller.

  Sound buttons (on press, not while held): Y → sad tune, A → victory tune.

Pure function (no hardware): tested in tests/test_policy.py.
"""

import math
from dataclasses import dataclass
from typing import Optional

import config
from drive import arcade_mix
from feet import FeetState, settle_speed
from ring import NONE, REVERSE
from whacker import HOME, STRIKE, whacker_speed
from xbox import XboxState


@dataclass(frozen=True)
class Inputs:
    pad: XboxState
    tape_hit: bool = False               # color sensor just landed on the tape this tick
    blocked_dir: int = NONE              # ring.REVERSE / NONE from EdgeGuard
    feet: Optional[FeetState] = None     # None = feet not calibrated → no settling
    opponent_entered: bool = False       # ProximityTrigger fired this tick (UNO Q distance)
    arm_phase: Optional[str] = None      # whacker.STRIKE / HOME move in progress, or None
    arm_target: Optional[int] = None     # ArmMotion's encoder target this tick
    arm_at_base: Optional[bool] = None   # None = arm position/geometry unknown
    manual_idle_s: float = math.inf      # seconds since the driver last used RT/LT
    pressed: frozenset = frozenset()     # buttons pressed THIS tick (xbox.ButtonEdges)

@dataclass(frozen=True)
class Action:
    left: int = 0
    right: int = 0
    reason: str = ""
    alert: bool = False                  # play edge sound + rumble the controller
    whack: int = 0                       # manual whacker speed (mode manual / off)
    whack_mode: str = "off"              # off / manual / strike / home / strike start / home start
    whack_target: Optional[int] = None   # encoder target to drive to (strike / home)
    arm_command: Optional[str] = None    # main.py: "strike" / "home" = start that move, "cancel"
    sound: Optional[str] = None          # "sad" / "victory": play this tune now

def decide_whacker(inputs):
    """Whacker sub-tree → dict of the Action's whacker fields."""
    pad = inputs.pad
    moving = inputs.arm_phase is not None

    def out(mode, whack=0, target=None, command=None):
        return dict(whack=whack, whack_mode=mode, whack_target=target, arm_command=command)

    if not pad.connected:
        return out("off", command="cancel" if moving else None)
    manual = whacker_speed(pad.rt, pad.lt)
    if manual != 0:
        return out("manual", whack=manual, command="cancel" if moving else None)
    if inputs.arm_phase == STRIKE:
        return out("strike", target=inputs.arm_target)
    if inputs.opponent_entered:
        return out("strike start", command=STRIKE)
    if inputs.arm_phase == HOME:
        return out("home", target=inputs.arm_target)
    if inputs.arm_at_base is False and inputs.manual_idle_s >= config.WHACKER_MANUAL_HOLD_S:
        return out("home start", command=HOME)
    return out("off")


SOUND_BUTTONS = {"y": "sad", "a": "victory"}


def decide_sound(inputs):
    """Y → sad tune, A → victory tune (if both on the same tick, sad wins)."""
    for button, tune in SOUND_BUTTONS.items():
        if button in inputs.pressed:
            return tune
    return None


def decide(inputs):
    pad = inputs.pad
    alert = inputs.tape_hit
    w = dict(alert=alert, sound=decide_sound(inputs), **decide_whacker(inputs))

    if not pad.connected:
        return Action(0, 0, "controller lost", **w)

    if pad.b:
        return Action(0, 0, "brake", **w)

    reverse_blocked = inputs.blocked_dir == REVERSE
    throttle = pad.ly
    reason = "teleop drive"
    if reverse_blocked and throttle < 0:
        throttle = 0.0
        reason = "edge: reverse blocked"

    left, right = arcade_mix(throttle=throttle, turn=pad.lx)
    if (left, right) != (0, 0) or inputs.feet is None:
        return Action(left, right, reason, **w)

    # Driver isn't asking for motion → roll each foot into its base arc.
    f = inputs.feet
    frac = config.SETTLE_TARGET_FRACTION
    left = settle_speed(f.left_pos, f.left_arc.middle(frac), f.left_dir, config.SETTLE_SPEED,
                        allow_reverse=not reverse_blocked)
    right = settle_speed(f.right_pos, f.right_arc.middle(frac), f.right_dir, config.SETTLE_SPEED,
                         allow_reverse=not reverse_blocked)
    if (left, right) == (0, 0):
        return Action(0, 0, "idle", **w)
    return Action(left, right, "settling feet", **w)
