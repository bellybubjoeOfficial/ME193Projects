"""Bounded FIFO of stamped samples waiting to be published.

When the broker is unreachable samples pile up here (capacity is set from
config buffer_seconds). If it ever fills, the OLDEST samples are dropped,
counted in `dropped`, and logged — never silently. The laptop will also
see the hole as a seq gap.
"""

import collections
import logging
import threading

log = logging.getLogger("ring")


class SampleRing:
    def __init__(self, capacity):
        self.capacity = capacity
        self.dropped = 0
        self._q = collections.deque()
        self._lock = threading.Lock()

    def __len__(self):
        return len(self._q)

    def extend(self, items):
        """items: (key, row) pairs. Rows sharing a key may go in one message."""
        with self._lock:
            self._q.extend(items)
            over = len(self._q) - self.capacity
            for _ in range(max(0, over)):
                self._q.popleft()
        if over > 0:
            self.dropped += over
            log.error("ring buffer full (%d samples): dropped %d oldest, %d total",
                      self.capacity, over, self.dropped)

    def take(self, n, current_key):
        """Pop up to n rows that share the first row's key.

        Returns (key, rows) or None. Normally only full batches of n come
        out; a shorter one is returned only once its key is finished (it is
        not `current_key`, i.e. the session ended or the rate changed).
        """
        with self._lock:
            if not self._q:
                return None
            key = self._q[0][0]
            rows = []
            for k, row in self._q:
                if k != key or len(rows) == n:
                    break
                rows.append(row)
            if len(rows) < n and key == current_key:
                return None
            for _ in rows:
                self._q.popleft()
            return key, rows

    def push_front(self, key, rows):
        """Put a batch back (publish was refused)."""
        with self._lock:
            self._q.extendleft((key, r) for r in reversed(rows))
