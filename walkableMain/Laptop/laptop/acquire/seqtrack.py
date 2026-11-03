"""Sequence-number bookkeeping: gaps, late (out-of-order) arrivals, duplicates.

One tracker per device session. A gap is reported the moment a jump in seq
is seen; if the missing samples show up later (out of order) they are
counted as `late` and credited back to that gap as `recovered`.
"""

from dataclasses import dataclass


@dataclass
class Gap:
    first_missing: int
    count: int
    t_detected_ns: int
    recovered: int = 0

    @property
    def last_missing(self):
        return self.first_missing + self.count - 1

    @property
    def still_missing(self):
        return self.count - self.recovered


class SeqTracker:
    def __init__(self):
        self.first = None
        self.next_expected = None
        self.received = 0     # unique samples
        self.duplicates = 0
        self.late = 0         # arrived after a later seq (filled a gap)
        self.missing = set()
        self.gaps = []

    def update(self, seqs, t_arrival_ns):
        """Feed one batch's seq numbers (in message order). Returns new Gaps."""
        new_gaps = []
        for s in (int(v) for v in seqs):
            if self.first is None:
                self.first, self.next_expected = s, s + 1
                self.received += 1
            elif s == self.next_expected:
                self.next_expected += 1
                self.received += 1
            elif s > self.next_expected:
                gap = Gap(self.next_expected, s - self.next_expected, t_arrival_ns)
                self.missing.update(range(self.next_expected, s))
                self.gaps.append(gap)
                new_gaps.append(gap)
                self.next_expected = s + 1
                self.received += 1
            elif s in self.missing:
                self.missing.remove(s)
                self._credit(s)
                self.late += 1
                self.received += 1
            elif s < self.first:
                # Out of order at the very start: the session began earlier.
                self.missing.update(range(s + 1, self.first))
                self.first = s
                self.late += 1
                self.received += 1
            else:
                self.duplicates += 1
        return new_gaps

    def _credit(self, s):
        for gap in reversed(self.gaps):
            if gap.first_missing <= s <= gap.last_missing:
                gap.recovered += 1
                return

    @property
    def expected(self):
        return 0 if self.first is None else self.next_expected - self.first

    @property
    def received_frac(self):
        return self.received / self.expected if self.expected else 0.0
