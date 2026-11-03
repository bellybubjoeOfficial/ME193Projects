"""Stage 3 -- heel-strike detector on synthetic heel trajectories."""

import numpy as np
import pytest

from laptop.config import load_config
from laptop.vision.steps import HeelStrikeDetector, StepDetector, params
from synth_gait import walk

P = params(load_config()["steps"], "heel_y")  # these trajectories model heel HEIGHT


def run(frames, t_ns, y, vis, p=P):
    det = HeelStrikeDetector("L", p)
    events = []
    for f, t, yy, vv in zip(frames, t_ns, y, vis):
        ev = det.update(int(f), int(t), None if np.isnan(yy) else float(yy), float(vv))
        if ev:
            events.append(ev)
    return events


def errors(events, truth):
    """Frame error of each detection vs its nearest true strike."""
    truth = np.array(truth)
    return [ev.frame_idx - truth[np.argmin(np.abs(truth - ev.frame_idx))] for ev in events]


def first_detectable(truth, W=P["smooth_frames"], n_frames=600):
    # The detector needs one full swing before it can report a strike, and a
    # few frames after it to confirm (so strikes in the clip's last frames can't count).
    return [s for s in truth if 1.5 * W < s < n_frames - W]


@pytest.mark.parametrize("gait", ["slow", "normal", "jog"])
@pytest.mark.parametrize("landing", ["hard", "soft"])
def test_counts_and_timing(gait, landing):
    frames, t, y, vis, truth = walk(gait, seconds=20, landing=landing, seed=1)
    W = P["smooth_frames"]
    ev = [e for e in run(frames, t, y, vis) if 1.5 * W < e.frame_idx < len(frames) - W]
    truth = first_detectable(truth)
    assert len(ev) == len(truth), (len(ev), len(truth))
    err = np.abs(errors(ev, truth))
    assert np.median(err) <= 1
    assert err.max() <= 2


@pytest.mark.parametrize("seed", range(5))
def test_noisy_landmarks_do_not_double_count(seed):
    frames, t, y, vis, truth = walk("normal", seconds=30, noise=0.012, seed=seed)
    ev = run(frames, t, y, vis)
    assert len(ev) == len(first_detectable(truth, n_frames=900))
    assert np.median(np.abs(errors(ev, truth))) <= 1


def test_standing_still_is_zero_steps():
    rng = np.random.default_rng(0)
    n = 600
    y = 3.0 + rng.normal(0, 0.01, n)
    t = (np.arange(n) / 30 * 1e9).astype(np.int64)
    assert run(np.arange(n), t, y, np.full(n, 0.9)) == []


def test_low_visibility_frames_are_ignored():
    frames, t, y, vis, _ = walk("normal", seconds=10)
    assert run(frames, t, y, np.full(len(y), 0.2)) == []


def test_short_dropout_keeps_the_step():
    frames, t, y, vis, truth = walk("normal", seconds=20, seed=2)
    clean = len(run(frames, t, y, vis))
    y2 = y.copy()
    for s in truth:          # hide 2 frames in mid-swing before every strike
        y2[max(0, s - 8):max(0, s - 6)] = np.nan
    assert len(run(frames, t, y2, vis)) == clean


def test_long_occlusion_never_fakes_a_strike():
    frames, t, y, vis, truth = walk("normal", seconds=20, seed=3)
    y2 = y.copy()
    y2[200:260] = np.nan     # far leg hidden behind the near leg for 2 s
    ev = run(frames, t, y2, vis)
    lost = [s for s in truth if 195 <= s <= 260 + 2 * P["smooth_frames"]]
    assert all(not (200 <= e.frame_idx < 260) for e in ev)
    assert len(ev) >= len(first_detectable(truth)) - len(lost) - 1


def test_bounce_after_landing_is_one_step():
    frames, t, y, vis, truth = walk("normal", seconds=12, seed=4, noise=0.003)
    y2 = y.copy()
    for s in truth:          # heel bounces 0.04 leg lengths 3-6 frames after landing
        if s + 6 < len(y2):
            y2[s + 3:s + 6] -= 0.04
    assert len(run(frames, t, y2, vis)) == len(run(frames, t, y, vis))


def test_15fps_camera_still_works():
    frames, t, y, vis, truth = walk("normal", seconds=20, fps=15, seed=5)
    ev = run(frames, t, y, vis)
    truth = [s for s in truth if 12 < s < len(frames) - 7]
    assert abs(len(ev) - len(truth)) <= 1
    assert np.median(np.abs(errors(ev, truth))) <= 1


def landmarks(y_heel_ll, leg_px, x0=600.0, vis=0.9):
    """Landmark dict for both legs from heel heights in leg lengths."""
    lm = {}
    for side, yh in zip(("left", "right"), y_heel_ll):
        hip_y = 100.0
        lm[f"{side}_hip"] = (x0, hip_y, vis)
        lm[f"{side}_ankle"] = (x0, hip_y + leg_px, vis)
        lm[f"{side}_heel"] = (x0 - 5, yh * leg_px, vis)
        lm[f"{side}_foot_index"] = (x0 + 30, yh * leg_px, vis)
    return lm


@pytest.mark.parametrize("leg_px", [120, 400])  # far from / near the camera
def test_both_legs_and_distance_invariance(leg_px):
    fL, t, yL, _, truthL = walk("normal", seconds=20, seed=6, phase0=0.3)
    _, _, yR, _, truthR = walk("normal", seconds=20, seed=7, phase0=0.8)
    det = StepDetector(P)
    for i in range(len(fL)):
        det.update(i, int(t[i]), landmarks((yL[i], yR[i]), leg_px))
    assert abs(det.counts["L"] - len(first_detectable(truthL))) <= 1
    assert abs(det.counts["R"] - len(first_detectable(truthR))) <= 1
    assert det.total == len(det.events)
    assert {e.leg for e in det.events} == {"L", "R"}


# ---- footprint method (default): leg-label agnostic ----
from laptop.vision.compare import compare, match_events  # noqa: E402
from laptop.vision.steps import FootprintDetector  # noqa: E402
from synth_gait import footprints_walk  # noqa: E402

FP = params(load_config()["steps"], "footprint")


def run_fp(frames, t_ns, lms, p=FP):
    det = FootprintDetector(p)
    for f, t, lm in zip(frames, t_ns, lms):
        det.update(int(f), int(t), lm)
    return det


@pytest.mark.parametrize("swap_prob", [0.0, 0.3])
def test_footprints_count_and_timing_even_with_label_swaps(swap_prob):
    frames, t, lms, truth = footprints_walk(seconds=20, swap_prob=swap_prob, seed=1)
    det = run_fp(frames, t, lms)
    truth = [s for s in truth if s[0] < len(frames) - FP["min_stance_frames"] - 2]
    pairs, missed, extra = match_events(sorted(e.frame_idx for e in det.events),
                                        sorted(f for f, _ in truth), 6)
    assert not missed and not extra
    assert np.median(np.abs([a - b for a, b in pairs])) <= 1


def test_footprint_legs_follow_majority_label():
    frames, t, lms, truth = footprints_walk(seconds=20, swap_prob=0.2, seed=2)
    det = run_fp(frames, t, lms)
    r = compare([(e.frame_idx, e.leg) for e in det.events], truth, 6)
    assert r["total"]["matched"] >= len(det.events) - 1  # labels right despite 20% swaps


def test_standing_still_makes_no_footprints():
    rng = np.random.default_rng(0)
    lms = [{"left_heel": (500 + rng.normal(0, 6), 600, 0.9), "right_heel": (520 + rng.normal(0, 6), 600, 0.9),
            "left_hip": (510, 370, 0.95), "right_hip": (510, 370, 0.95),
            "left_ankle": (505, 590, 0.9), "right_ankle": (525, 590, 0.9)} for _ in range(300)]
    t = (np.arange(300) / 30 * 1e9).astype(np.int64)
    assert run_fp(np.arange(300), t, lms).events == []
