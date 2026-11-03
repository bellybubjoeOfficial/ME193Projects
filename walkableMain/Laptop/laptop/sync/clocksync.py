"""NTP-style clock offset between the UNO Q and the laptop.

    t0 = laptop sends ping        t1 = device receives it
    t2 = device sends pong        t3 = laptop receives pong

    offset = ((t1 - t0) + (t2 - t3)) / 2     device clock minus laptop clock
    rtt    = (t3 - t0) - (t2 - t1)           network time, excluding device hold

This assumes the two network legs take equal time, so each single estimate
can be off by up to rtt/2. The running median over many pings removes the
spikes; pings with a huge rtt are dropped from the median entirely.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np


def ntp_offset_rtt(t0, t1, t2, t3):
    offset = ((t1 - t0) + (t2 - t3)) // 2
    rtt = (t3 - t0) - (t2 - t1)
    return offset, rtt


@dataclass
class SyncSample:
    t_laptop_ns: int   # t3, when we learned it
    offset_ns: int
    rtt_ns: int
    accepted: bool


class ClockSync:
    def __init__(self, median_window=31, max_rtt_ms=250):
        self.max_rtt_ns = int(max_rtt_ms * 1e6)
        self._recent = deque(maxlen=median_window)  # accepted samples only
        self.history = []                           # every sample, for logs/drift
        self._offset_ns = None

    def add(self, t0, t1, t2, t3):
        offset, rtt = ntp_offset_rtt(t0, t1, t2, t3)
        accepted = 0 <= rtt <= self.max_rtt_ns
        sample = SyncSample(t3, offset, rtt, accepted)
        self.history.append(sample)
        if accepted:
            self._recent.append(sample)
            self._offset_ns = int(np.median([s.offset_ns for s in self._recent]))
        return sample

    @property
    def offset_ns(self):
        """Running median offset, or None before the first good pong."""
        return self._offset_ns

    @property
    def rtt_ns(self):
        if not self._recent:
            return None
        return int(np.median([s.rtt_ns for s in self._recent]))

    @property
    def synced(self):
        return self._offset_ns is not None

    def device_to_laptop(self, t_dev_ns):
        """Works on ints or int64 arrays. Requires synced."""
        return t_dev_ns - self._offset_ns

    def drift_ppm(self, min_span_s=120):
        """Slope of offset vs. laptop time (device clock gain, ppm).

        Offset noise is ~ms, so this is meaningless over less than a few minutes.
        """
        good = [s for s in self.history if s.accepted]
        if len(good) < 3:
            return None
        t = np.array([s.t_laptop_ns for s in good], dtype=np.float64)
        off = np.array([s.offset_ns for s in good], dtype=np.float64)
        if t[-1] - t[0] < min_span_s * 1e9:
            return None
        slope = np.polyfit(t - t[0], off - off[0], 1)[0]
        return slope * 1e6
