"""
drive.py

Stick → wheel speeds (pure math, unit tested in tests/test_drive.py).
"""

import config


def arcade_mix(throttle, turn, max_speed=None, turn_gain=None):
    """Arcade drive: one stick, up/down = throttle, left/right = turn.

    throttle, turn: -1..1 (turn +1 = turn right)
    Returns (left, right) integer speeds in -max_speed..max_speed. If the
    mix overshoots, both sides are scaled down together so the turn
    ratio is preserved instead of one side clipping.
    """
    max_speed = config.MAX_SPEED if max_speed is None else max_speed
    turn_gain = config.TURN_GAIN if turn_gain is None else turn_gain

    turn = turn * turn_gain
    left = throttle + turn
    right = throttle - turn
    scale = max(1.0, abs(left), abs(right))
    return round(left / scale * max_speed), round(right / scale * max_speed)


def apply_inversion(left, right):
    """Flip either side if it's mounted backwards (see config.INVERT_*)."""
    return (-left if config.INVERT_LEFT else left,
            -right if config.INVERT_RIGHT else right)
