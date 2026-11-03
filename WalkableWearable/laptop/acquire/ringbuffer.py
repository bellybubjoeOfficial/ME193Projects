"""Thread-safe fixed-size ring of IMU samples (MQTT thread writes, UI reads)."""

import threading

import numpy as np


class RingBuffer:
    def __init__(self, capacity, n_channels=6):
        self.capacity = capacity
        self._t = np.zeros(capacity, dtype=np.int64)
        self._seq = np.zeros(capacity, dtype=np.int64)
        self._data = np.zeros((capacity, n_channels), dtype=np.float64)
        self._head = 0    # next write index
        self._count = 0
        self._lock = threading.Lock()

    def __len__(self):
        with self._lock:
            return self._count

    def clear(self):
        with self._lock:
            self._head = self._count = 0

    def append(self, t_ns, seq, data):
        n = len(t_ns)
        if n > self.capacity:  # only the newest `capacity` samples can fit
            t_ns, seq, data = t_ns[-self.capacity:], seq[-self.capacity:], data[-self.capacity:]
            n = self.capacity
        with self._lock:
            idx = (self._head + np.arange(n)) % self.capacity
            self._t[idx] = t_ns
            self._seq[idx] = seq
            self._data[idx] = data
            self._head = (self._head + n) % self.capacity
            self._count = min(self._count + n, self.capacity)

    def snapshot(self, since_ns=None):
        """Copies of (t_ns, seq, data), sorted by time, optionally t >= since_ns."""
        with self._lock:
            start = (self._head - self._count) % self.capacity
            idx = (start + np.arange(self._count)) % self.capacity
            t, seq, data = self._t[idx], self._seq[idx], self._data[idx]
        # Late (out-of-order) samples were appended late; put them in place.
        order = np.argsort(t, kind="stable")
        t, seq, data = t[order], seq[order], data[order]
        if since_ns is not None:
            keep = t >= since_ns
            t, seq, data = t[keep], seq[keep], data[keep]
        return t, seq, data
