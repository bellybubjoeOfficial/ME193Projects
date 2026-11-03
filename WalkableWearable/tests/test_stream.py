"""Stage 7 -- streaming (live) detection matches offline, with bounded latency."""

import numpy as np
import torch
from torch import nn

from laptop.config import load_config
from laptop.train.dataset import Normalizer
from laptop.train.events import BoutGate, bout_filter, predict_events
from laptop.train.model import predict
from laptop.train.stream import StreamingDetector, stream_arrays
from laptop.vision.compare import match_events

CFG = load_config()


class SpikeModel(nn.Module):
    """Stand-in CNN: a smoothed az spike -> L, ay spike -> R. Same receptive field as StepCNN."""
    receptive_field = 67

    def __init__(self):
        super().__init__()
        k = torch.exp(-0.5 * (torch.arange(-6, 7) / 2.0) ** 2)
        self.register_buffer("k", (k / k.sum()).view(1, 1, -1))

    def forward(self, x):
        z = nn.functional.conv1d(x[:, 2:3], self.k, padding=6)
        y = nn.functional.conv1d(x[:, 1:2], self.k, padding=6)
        return torch.cat([4 * z - 6, 4 * y - 6], dim=1)


def session(seconds=60, seed=0, drop=()):
    rng = np.random.default_rng(seed)
    n = seconds * 100
    x = rng.normal(0, 0.2, (n, 6)).astype(np.float32)
    strikes = []
    for k, ts in enumerate(np.arange(1.0, seconds - 1, 0.55)):
        i = int(ts * 100)
        x[i:i + 3, 2 if k % 2 == 0 else 1] += 6
        strikes.append((i, "L" if k % 2 == 0 else "R"))
    seq = np.arange(n) + 5000
    t = (np.arange(n) * 1e7).astype(np.int64) + 10**12
    keep = np.ones(n, bool)
    keep[list(drop)] = False
    return t[keep], seq[keep], x[keep], strikes


def detector(lookahead_s=None, bout_min_steps=None):
    """Default None = exact lookahead, so live must equal offline sample for sample."""
    live = {**CFG["live"], "lookahead_s": lookahead_s}
    if bout_min_steps is not None:
        live["bout_min_steps"] = bout_min_steps
    cfg = {**CFG, "live": live}
    return StreamingDetector(SpikeModel().eval(), Normalizer(np.zeros(6), np.ones(6)), cfg)


def test_streaming_matches_offline():
    t, seq, x, _ = session()
    det = detector()
    live = stream_arrays(det, t, seq, x)
    probs = predict(det.model, x)
    offline = predict_events(probs, 100, CFG["train"]["threshold"], CFG["train"]["refractory_s"])
    lc = CFG["live"]
    offline = bout_filter(offline, t, lc["bout_min_steps"], lc["bout_max_gap_s"], lc["bout_need_both_legs"])
    live_idx = sorted((e.seq - seq[0], e.leg) for e in live)
    for leg in "LR":
        a = [i for i, l in live_idx if l == leg]
        b = [i for i, l in offline if l == leg and i < len(seq) - det.lookahead - 1]
        pairs, missed, extra = match_events(a, b, 0)  # identical sample, not just close
        assert not missed and not extra


def test_each_step_reported_once_and_counted():
    t, seq, x, strikes = session(seconds=30)
    det = detector()
    live = stream_arrays(det, t, seq, x)
    assert len({(e.seq, e.leg) for e in live}) == len(live)
    assert det.total == len(live) == len([s for s in strikes if s[0] < len(seq) - det.lookahead - 1])
    assert det.counts["L"] > 0 and det.counts["R"] > 0


def test_latency_is_lookahead_plus_batching():
    t, seq, x, _ = session(seconds=30)
    det = detector()
    # The first steps of a bout are counted retroactively (late by design); check the rest.
    steps = stream_arrays(det, t, seq, x, chunk=10, latency_ms=36)
    lat = np.array([e.latency_ms for e in steps[CFG["live"]["bout_min_steps"]:]])
    look_ms = det.lookahead * 10
    assert lat.min() >= look_ms + 36 - 1
    assert lat.max() <= look_ms + 36 + 100 + 1   # + at most one 10-sample batch


def test_lost_samples_do_not_break_or_duplicate():
    t, seq, x, strikes = session(seconds=30, drop=range(700, 760))
    det = detector()
    live = stream_arrays(det, t, seq, x)
    assert len({(e.seq, e.leg) for e in live}) == len(live)
    assert det.total >= len(strikes) - 3   # only steps inside the 0.6 s hole can be lost


def test_cadence():
    t, seq, x, _ = session(seconds=30)
    det = detector()
    stream_arrays(det, t, seq, x)
    # one strike every 0.55 s = 109 steps/min
    assert abs(det.cadence_spm(int(t[-1])) - 60 / 0.55) < 12


def test_device_restart_resets():
    t, seq, x, _ = session(seconds=20)
    det = detector()
    stream_arrays(det, t, seq, x)
    assert det.total > 0
    before = det.total
    det.update(t[:500], seq[:500] - 5000, x[:500])  # seq jumps back to ~0
    assert det.total < before and all(e.seq < 500 for e in det.events)


def test_shorter_lookahead_is_faster_and_still_finds_steps():
    t, seq, x, strikes = session(seconds=30)
    full, fast = detector(None), detector(0.15)
    a, b = stream_arrays(full, t, seq, x), stream_arrays(fast, t, seq, x)
    assert fast.lookahead < full.lookahead
    assert np.median([e.latency_ms for e in b]) < np.median([e.latency_ms for e in a]) - 100
    pairs, missed, extra = match_events(sorted(e.seq for e in b), sorted(e.seq for e in a), 3)
    assert len(missed) <= 1 and len(extra) <= 1


def test_lookahead_never_exceeds_exact():
    assert detector(5.0).lookahead == detector(None).lookahead


# ---- walking-bout gate ----
def run_gate(times_legs, **kw):
    g = BoutGate(**kw)
    out = []
    for t, leg in times_legs:
        out += g.push(int(t * 1e9), leg, (t, leg))
    return out


def test_isolated_twitches_never_count():
    twitches = [(t, "R") for t in np.arange(0, 60, 3.0)] + [(t, "L") for t in np.arange(1.5, 60, 4.0)]
    assert run_gate(sorted(twitches), min_steps=4, max_gap_s=1.5) == []


def test_one_foot_tapping_never_counts():
    taps = [(t, "R") for t in np.arange(0, 20, 0.5)]   # rhythmic, but one leg
    assert run_gate(taps, min_steps=4, max_gap_s=1.5, need_both_legs=True) == []


def test_walking_bout_counts_every_step_including_the_first():
    walk = [(1 + 0.55 * k, "L" if k % 2 == 0 else "R") for k in range(20)]
    assert run_gate(walk, min_steps=4, max_gap_s=1.5) == walk


def test_pause_then_new_bout():
    a = [(0.55 * k, "LR"[k % 2]) for k in range(6)]
    b = [(10 + 0.55 * k, "LR"[k % 2]) for k in range(6)]
    lone = [(6.0, "L")]                                  # a shuffle during the pause
    assert run_gate(a + lone + b, min_steps=4, max_gap_s=1.5) == a + b


def test_gate_off():
    assert run_gate([(0, "R"), (5, "R")], min_steps=1) == [(0, "R"), (5, "R")]


def test_live_ignores_fidgeting():
    """With the gate on: isolated az spikes 3 s apart (ankle twitches) -> zero steps live."""
    rng = np.random.default_rng(1)
    n = 3000
    x = rng.normal(0, 0.2, (n, 6)).astype(np.float32)
    for ts in np.arange(2, 28, 3.0):
        x[int(ts * 100):int(ts * 100) + 3, 2] += 6
    det = detector(bout_min_steps=4)
    live = stream_arrays(det, (np.arange(n) * 1e7).astype(np.int64), np.arange(n), x)
    assert live == [] and det.total == 0
