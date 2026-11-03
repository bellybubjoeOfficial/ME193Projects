"""MCU clock -> Linux CLOCK_MONOTONIC mapping.

Every sample is stamped on the MCU (64-bit cycle counter, ns) right before
the I2C read. To turn that into Linux time we periodically ask the MCU for
its clock (Bridge call `now_ns`) and note the Linux monotonic time before
(t0) and after (t1) the call. The MCU read happened somewhere inside
[t0, t1], so the midpoint is the estimate and the error is at most RTT/2.

The Bridge UART is shared with the sample batches, so some calls queue
behind a batch (RTT 20+ ms). We keep only the lowest-RTT point per
`bucket_s` and fit  t_linux = t_mcu + offset + skew * (t_mcu - ref)  over
the last `window_s`: offset absorbs the boot-time difference, skew the
oscillator drift. On the UNO Q this was measured at ~+880 ppm (the MCU
clock runs slow, ~0.5 s per 10 min), so it matters a lot. `max_skew_ppm`
only rejects nonsense fits; it must stay well above the real drift.

Samples are mapped with `map_sample`, not the raw fit: each refit moves the
fit by up to a few ms (by ~20 ms when drift is first estimated), and
stepping sample times by that much would corrupt the intervals. Instead the
mapping slews toward the fit by at most `max_slew_ppm` of each interval, so
t_mono stays continuous and an error of 20 ms is absorbed in ~4 s.
"""

import threading


class ClockMap:
    def __init__(self, window_s=600, bucket_s=10, min_span_for_skew_s=20, max_skew_ppm=10_000,
                 max_slew_ppm=5_000, max_hold_s=5):
        self.window_ns = int(window_s * 1e9)
        self.bucket_ns = int(bucket_s * 1e9)
        self.min_span_ns = int(min_span_for_skew_s * 1e9)
        self.max_skew = max_skew_ppm * 1e-6
        self.max_slew = max_slew_ppm * 1e-6
        self.max_hold_ns = int(max_hold_s * 1e9)
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        """Forget everything (the MCU rebooted: its clock restarted from 0)."""
        with self._lock:
            self._best = {}  # bucket index -> (rtt, t_mcu, offset)
            self._fit = None  # (ref_mcu, offset_at_ref, skew)
            self._last_mcu = None
            self.last_rtt_ns = None
            self._out = None  # (t_mcu, t_linux) of the last mapped sample

    @property
    def ready(self):
        return self._fit is not None

    def add(self, t0_linux, t_mcu, t1_linux):
        """Add one sync exchange. Returns False if it showed an MCU reboot."""
        rtt = t1_linux - t0_linux
        offset = (t0_linux + t1_linux) // 2 - t_mcu
        with self._lock:
            if self._last_mcu is not None and t_mcu < self._last_mcu:
                return False
            self._last_mcu = t_mcu
            self.last_rtt_ns = rtt
            b = t_mcu // self.bucket_ns
            if b not in self._best or rtt < self._best[b][0]:
                self._best[b] = (rtt, t_mcu, offset)
            oldest = (t_mcu - self.window_ns) // self.bucket_ns
            for k in [k for k in self._best if k < oldest]:
                del self._best[k]
            self._refit()
        return True

    def _refit(self):
        pts = sorted(self._best.values(), key=lambda p: p[1])
        ref = pts[-1][1]
        span = ref - pts[0][1]
        if len(pts) < 3 or span < self.min_span_ns:
            # Too short to see drift: use the single best (lowest RTT) point.
            best = min(pts)
            self._fit = (best[1], best[2], 0.0)
            return
        # Least squares offset = a + skew * x, x relative to the newest point.
        xs = [p[1] - ref for p in pts]
        ys = [p[2] for p in pts]
        n = len(pts)
        mx, my = sum(xs) / n, sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        skew = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
        skew = max(-self.max_skew, min(self.max_skew, skew))
        self._fit = (ref, my - skew * mx, skew)

    def to_linux(self, t_mcu):
        """The current fit, as is (may step when the fit is updated)."""
        ref, off, skew = self._fit
        return t_mcu + int(off + skew * (t_mcu - ref))

    def map_sample(self, t_mcu):
        """Map a sample time, continuous with the previous one (call in order).

        Follows the fit's rate, and closes any difference to the fit by at
        most max_slew of the elapsed time. After a pause longer than
        max_hold_s (stream stopped) it jumps straight to the fit.
        """
        target = self.to_linux(t_mcu)
        prev = self._out
        if prev is None or not 0 < t_mcu - prev[0] <= self.max_hold_ns:
            out = target
        else:
            dt = t_mcu - prev[0]
            pred = prev[1] + int(dt * (1 + self._fit[2]))
            lim = int(dt * self.max_slew)
            out = pred + max(-lim, min(lim, target - pred))
        self._out = (t_mcu, out)
        return out

    def info(self):
        with self._lock:
            if self._fit is None:
                return {"ready": False}
            best = min(self._best.values()) if self._best else None
            return {
                "ready": True,
                "skew_ppm": round(self._fit[2] * 1e6, 2),
                "points": len(self._best),
                "best_rtt_ms": round(best[0] / 1e6, 3) if best else None,
                "last_rtt_ms": round(self.last_rtt_ns / 1e6, 3) if self.last_rtt_ns else None,
            }
