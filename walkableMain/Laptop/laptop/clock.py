"""The one laptop clock every stage stamps with.

time.monotonic_ns() never jumps (no NTP corrections) and on macOS it is
system-wide, so the IMU process and the webcam process share it.
"""

import time


def now_ns():
    return time.monotonic_ns()


def wall_ns():
    """Wall-clock time, only for human-readable labels in metadata."""
    return time.time_ns()
