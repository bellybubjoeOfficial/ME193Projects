"""
ring.py

Downward color sensor → "are we on the blue ring tape?", plus the short
memory the decision tree needs about it.

EdgeGuard blocks REVERSE while the sensor sees tape and for EDGE_BLOCK_S
seconds after the last tape reading. Forward and turning are always
allowed.

Pure logic (time is passed in): tested in tests/test_ring.py.
"""

import config

REVERSE = -1
NONE = 0


def is_tape(color_name):
    """True if the color sensor's named color counts as ring tape."""
    return color_name in config.TAPE_COLOR_NAMES


class EdgeGuard:
    def __init__(self, block_s=None):
        self.block_s = config.EDGE_BLOCK_S if block_s is None else block_s
        self._was_on_tape = False
        self._blocked_until = float("-inf")

    def update(self, on_tape, now):
        """Feed one sensor reading (`now` = time.monotonic() seconds).

        Every tape reading pushes the reverse block out to now + block_s.
        Returns True only on the tick the sensor first lands on the tape
        (rising edge) -- that's when the alert should fire.
        """
        hit = on_tape and not self._was_on_tape
        self._was_on_tape = on_tape
        if on_tape:
            self._blocked_until = now + self.block_s
        return hit

    def blocked_direction(self, now):
        """REVERSE while the block is active, else NONE."""
        return REVERSE if now < self._blocked_until else NONE
