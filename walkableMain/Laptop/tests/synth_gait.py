"""Synthetic side-on heel trajectories with known heel-strike times.

Heel height above the ground h(phase), phase 0 = heel strike:
  stance  (phase < stance):  flat, then the heel peels up in late stance
  swing:                     arc of height `swing_h`, landing at phase 1
`landing="hard"` hits the ground still moving down (sin arc);
`landing="soft"` slows to zero first (sin^2 arc).
Image y = ground - h (y down), all in leg lengths.
"""

import numpy as np

GAITS = {  # period s, stance fraction, swing height (leg lengths), heel-off rise
    "slow": dict(period=1.35, stance=0.65, swing_h=0.12, heel_rise=0.05),
    "normal": dict(period=1.05, stance=0.60, swing_h=0.20, heel_rise=0.08),
    "jog": dict(period=0.72, stance=0.40, swing_h=0.32, heel_rise=0.10),
}


def heel_height(phase, period, stance, swing_h, heel_rise, landing="hard"):
    p = np.mod(phase, 1.0)
    h = np.zeros_like(p)
    peel_start = 0.6 * stance
    peel = (p >= peel_start) & (p < stance)
    h[peel] = heel_rise * ((p[peel] - peel_start) / (stance - peel_start)) ** 2
    sw = p >= stance
    q = (p[sw] - stance) / (1 - stance)
    arc = np.sin(np.pi * q) if landing == "hard" else np.sin(np.pi * q) ** 2
    h[sw] = swing_h * arc + heel_rise * (1 - q) ** 2
    return h


def walk(gait="normal", seconds=20.0, fps=30.0, noise=0.006, landing="hard", seed=0,
         fps_jitter_ms=1.5, ground=3.0, phase0=0.3, vis=0.95):
    """Returns (frame_idx, t_ns, y, vis, strike_frames) for one leg."""
    g = GAITS[gait]
    rng = np.random.default_rng(seed)
    n = int(seconds * fps)
    frames = np.arange(n)
    t = frames / fps + rng.normal(0, fps_jitter_ms / 1000, n)
    t = np.maximum.accumulate(t)  # timestamps never go backwards
    phase = t / g["period"] + phase0
    y = ground - heel_height(phase, g["period"], g["stance"], g["swing_h"], g["heel_rise"], landing)
    y = y + rng.normal(0, noise, n)
    strike_times = (np.arange(np.ceil(phase[0]), np.floor(phase[-1]) + 1) - phase0) * g["period"]
    strike_frames = [int(np.argmin(np.abs(t - ts))) for ts in strike_times
                     if t[0] < ts < t[-1]]
    t_ns = (t * 1e9).astype(np.int64) + 10**12
    return frames, t_ns, y, np.full(n, vis), strike_frames


def footprints_walk(seconds=20.0, fps=30.0, period=1.05, stance=0.6, step_px=180.0,
                    leg_px=230.0, noise_px=6.0, swap_prob=0.0, seed=0, x0=200.0):
    """Two heels in pixels, walking right; MediaPipe-style L/R label swaps.

    Each leg: stance = heel parked on its footprint; swing = heel slides to a
    footprint 2 steps ahead (smooth-step). Legs half a period apart.
    Returns (frames, t_ns, landmarks list, strikes [(frame, leg)]).
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * fps)
    t = np.arange(n) / fps
    heel = {}
    strikes = []
    for leg, offset in (("L", 0.0), ("R", 0.5)):
        cyc = t / period + offset
        k, ph = np.floor(cyc), cyc % 1.0
        base = x0 + (2 * k + (1 if leg == "R" else 0)) * step_px
        q = np.clip((ph - stance) / (1 - stance), 0, 1)
        x = base + 2 * step_px * (3 * q**2 - 2 * q**3)
        y = 600 - 40 * np.sin(np.pi * q)       # small lift (floor-level camera)
        heel[leg] = (x + rng.normal(0, noise_px, n), y + rng.normal(0, noise_px, n))
        for kk in range(int(k.min()) + 1, int(k.max()) + 1):
            f = int(round((kk - offset) * period * fps))
            if 0 < f < n:
                strikes.append((f, leg))
    lms = []
    for i in range(n):
        lm = {}
        legs = {"L": "left", "R": "right"}
        if rng.random() < swap_prob:  # MediaPipe swaps which heel is "left"
            legs = {"L": "right", "R": "left"}
        hip_x = (heel["L"][0][i] + heel["R"][0][i]) / 2
        for leg, side in legs.items():
            lm[f"{side}_heel"] = (heel[leg][0][i], heel[leg][1][i], 0.9)
            lm[f"{side}_hip"] = (hip_x, 600 - leg_px, 0.95)
            lm[f"{side}_ankle"] = (heel[leg][0][i] + 10, heel[leg][1][i] - 10, 0.9)
        lms.append(lm)
    t_ns = (t * 1e9).astype(np.int64) + 10**12
    return np.arange(n), t_ns, lms, sorted(strikes)
