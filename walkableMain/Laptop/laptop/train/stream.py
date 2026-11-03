"""Stage 7: run the step CNN on a live IMU stream.

On every update the model runs over the last `live.buffer_s` of IMU. A
sample's probability is only exactly FINAL once the model has seen half its
receptive field after it (the CNN is centred: 0.64 s). In practice it
barely changes after ~0.3 s, so steps are called `live.lookahead_s` after
they happen (set None for exact-offline behaviour), plus batching and
network delay. Provisional probabilities for the newest samples are still
returned for display.

Peaks are picked in time order on final probabilities: a local maximum
above `threshold`, at least `refractory_s` after the previous step on that
leg. They are then only counted inside walking bouts (events.BoutGate), so
fidgeting doesn't count. Steps are tracked by IMU seq, so overlapping
buffers never report a step twice.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np

from laptop.train.events import BoutGate
from laptop.train.model import predict
from laptop.vision.steps import LEGS


@dataclass
class LiveStep:
    t_ns: int        # when the heel struck (IMU laptop time)
    leg: str
    prob: float
    seq: int
    latency_ms: float  # reported this long after the strike


class StreamingDetector:
    def __init__(self, model, norm, cfg, rate_hz=100.0, device="cpu"):
        self.model, self.norm, self.device = model, norm, device
        self.rate = float(rate_hz)
        self.buffer_n = int(cfg["live"]["buffer_s"] * self.rate)
        self.full_lookahead = model.receptive_field // 2 + 1
        self.set_lookahead(cfg["live"].get("lookahead_s"))
        self.threshold = cfg["train"]["threshold"]
        self.refractory_s = cfg["train"]["refractory_s"]
        self.cadence_window_s = cfg["live"]["cadence_window_s"]
        lc = cfg["live"]
        self.gate = BoutGate(lc.get("bout_min_steps", 1), lc.get("bout_max_gap_s", 1.5),
                             lc.get("bout_need_both_legs", False))
        self.reset()

    def reset(self):
        self.events = []
        self.counts = {leg: 0 for leg in LEGS}
        self._last_seq = {leg: None for leg in LEGS}
        self._scan_from = None   # first seq not yet checked for peaks
        self._recent = deque()   # step times for cadence
        self.view = None         # (t_grid, probs, final_end) for plotting
        if hasattr(self, "gate"):
            self.gate.reset()

    def set_lookahead(self, seconds):
        """None = exact (half the receptive field); never more than that."""
        n = self.full_lookahead if seconds is None else int(round(seconds * self.rate))
        self.lookahead = max(2, min(n, self.full_lookahead))

    @property
    def latency_s(self):
        return self.lookahead / self.rate

    @property
    def total(self):
        return sum(self.counts.values())

    def update(self, t_ns, seq, data, now_ns=None):
        """t_ns/seq/data: recent IMU samples (any order, gaps allowed). Returns new LiveSteps."""
        seq = np.asarray(seq, dtype=np.int64)
        if len(seq) < 2 * self.lookahead:
            return []
        order = np.argsort(seq)
        seq, t_ns, data = seq[order], np.asarray(t_ns, dtype=np.int64)[order], np.asarray(data)[order]
        s1 = int(seq[-1])
        if self._scan_from is not None and s1 < self._scan_from - self.buffer_n:
            self.reset()  # seq went backwards: the device restarted
        s0 = max(int(seq[0]), s1 - self.buffer_n + 1)
        keep = seq >= s0
        seq, t_ns, data = seq[keep], t_ns[keep], data[keep]
        n = s1 - s0 + 1
        idx = seq - s0
        x = np.zeros((n, data.shape[1]), dtype=np.float32)
        x[idx] = data
        valid = np.zeros(n, dtype=bool)
        valid[idx] = True
        t_grid = np.interp(np.arange(n), idx, t_ns.astype(np.float64)).astype(np.int64)

        probs = predict(self.model, self.norm(x) * valid[:, None], self.device)
        final_end = n - 1 - self.lookahead   # last index whose probability won't change
        self.view = (t_grid, probs, final_end)
        if final_end < 1:
            return []

        start = 1 if self._scan_from is None else max(1, self._scan_from - s0)
        now = int(now_ns if now_ns is not None else t_grid[-1])
        refr = self.refractory_s * self.rate
        new = []
        for i in range(start, final_end):        # i+1 must be final too (local-max test)
            for k, leg in enumerate(LEGS):
                p = probs[:, k]
                if not (p[i] >= self.threshold and p[i] >= p[i - 1] and p[i] > p[i + 1]):
                    continue
                s = s0 + i
                if self._last_seq[leg] is not None and s - self._last_seq[leg] < refr:
                    continue
                self._last_seq[leg] = s
                cand = LiveStep(int(t_grid[i]), leg, float(p[i]), s, 0.0)
                for ev in self.gate.push(cand.t_ns, leg, cand):   # walking bouts only
                    ev.latency_ms = (now - ev.t_ns) / 1e6
                    self.events.append(ev)
                    self.counts[ev.leg] += 1
                    self._recent.append(ev.t_ns)
                    new.append(ev)
        self._scan_from = s0 + final_end
        return new

    def cadence_spm(self, now_ns):
        """Steps per minute from the median gap between steps (either leg) in the
        last cadence_window_s. Gaps, not counts: the newest ~0.35 s isn't final yet."""
        w = self.cadence_window_s * 1e9
        while self._recent and self._recent[0] < now_ns - w:
            self._recent.popleft()
        if len(self._recent) < 3:
            return 0.0
        return 60e9 / float(np.median(np.diff(sorted(self._recent))))


def stream_arrays(detector, t_ns, seq, x, chunk=10, latency_ms=36.0):
    """Feed a recorded session through the detector in MQTT-sized chunks (replay)."""
    out = []
    for end in range(chunk, len(seq) + 1, chunk):
        a = max(0, end - detector.buffer_n)
        now = int(t_ns[end - 1] + latency_ms * 1e6)
        out += detector.update(t_ns[a:end], seq[a:end], x[a:end], now_ns=now)
    return out
