"""U3.2 / U4.2 -- RT / LT → whacker speed, proximity trigger, one-whack sequencer."""

import unittest

import config
import json
import math
import os
import tempfile

from feet import Arc
from whacker import (HOME, STRIKE, ArmGeometry, ArmMotion, ProximityTrigger, SweepSample,
                     blind_arcs_from_sweep, center_relative, in_blind_zone, load_blind_calibration,
                     load_center_abs, manual_move, save_blind_calibration, visible_distance,
                     whacker_speed)


class TestWhackerSpeed(unittest.TestCase):
    def test_released_is_zero(self):
        self.assertEqual(whacker_speed(0, 0, max_speed=100), 0)

    def test_rt_and_lt_go_opposite_ways(self):
        self.assertEqual(whacker_speed(1, 0, max_speed=100), 100)
        self.assertEqual(whacker_speed(0, 1, max_speed=100), -100)

    def test_proportional(self):
        self.assertEqual(whacker_speed(0.5, 0, max_speed=100), 50)

    def test_both_pulled_cancel(self):
        self.assertEqual(whacker_speed(1, 1, max_speed=100), 0)


class TestProximityTrigger(unittest.TestCase):
    def setUp(self):
        self.p = ProximityTrigger(trigger_mm=200, rearm_mm=250)

    def test_default_threshold_is_the_config_constant(self):
        self.assertEqual(ProximityTrigger().trigger_mm, config.WHACK_TRIGGER_MM)

    def test_far_does_nothing(self):
        self.assertFalse(any(self.p.update(mm) for mm in (800, 500, 201)))

    def test_fires_once_on_entering_range(self):
        self.assertFalse(self.p.update(300))
        self.assertTrue(self.p.update(190))
        self.assertFalse(self.p.update(150))
        self.assertFalse(self.p.update(120))

    def test_exactly_200_counts(self):
        self.assertTrue(self.p.update(200))

    def test_noise_around_threshold_does_not_retrigger(self):
        self.p.update(190)
        self.assertFalse(any(self.p.update(mm) for mm in (205, 195, 230, 199, 249)))

    def test_rearms_after_backing_off(self):
        self.p.update(190)
        self.p.update(260)               # beyond 250 → re-armed
        self.assertTrue(self.p.update(180))

    def test_unknown_distance_never_fires_or_rearms(self):
        self.assertFalse(self.p.update(None))
        self.p.update(190)
        self.p.update(None)
        self.assertFalse(self.p.update(150), "None must not count as backing off")


class FakeArm:
    """Motor hub with onboard position control: each tick moves up to
    `rate` degrees toward the commanded target and stops on it.
    `jam_at`: relative positions beyond which the arm can't move (lo, hi)."""

    def __init__(self, start=0.0, rate=20.0, jam=None):
        self.pos = start
        self.rate = rate
        self.jam = jam

    def drive_to(self, target):
        if target is None:
            return
        new = self.pos + max(-self.rate, min(self.rate, target - self.pos))
        if self.jam is not None and not (self.jam[0] <= new <= self.jam[1]):
            return
        self.pos = new


class TestArmGeometry(unittest.TestCase):
    def test_base_and_strike_end(self):
        g = ArmGeometry(center=1000, base_side=1, base_deg=90, strike_deg=90)
        self.assertEqual((g.base, g.strike_end), (1090, 910))

    def test_base_on_the_other_side(self):
        g = ArmGeometry(center=1000, base_side=-1, base_deg=90, strike_deg=90)
        self.assertEqual((g.base, g.strike_end), (910, 1090))

    def test_defaults_come_from_config(self):
        g = ArmGeometry(center=0)
        self.assertEqual(g.base, round(config.WHACKER_BASE_SIDE * config.WHACKER_BASE_DEG))
        self.assertEqual(g.strike_end, round(-config.WHACKER_BASE_SIDE * config.WHACKER_STRIKE_DEG))

    def test_at_base(self):
        g = ArmGeometry(center=0, base_side=1, base_deg=90, strike_deg=90)
        self.assertTrue(g.at_base(92, tol=5))
        self.assertFalse(g.at_base(80, tol=5))


class TestCenterRelative(unittest.TestCase):
    def test_arm_at_center(self):
        self.assertEqual(center_relative(rel_now=500, abs_now=275, center_abs=275), 500)

    def test_arm_off_center(self):
        # Arm is 90° past center (abs 5 vs center 275, across 0°).
        self.assertEqual(center_relative(rel_now=590, abs_now=5, center_abs=275), 500)
        self.assertEqual(center_relative(rel_now=410, abs_now=185, center_abs=275), 500)

    def test_counter_reset_after_power_cycle(self):
        # Same physical arm angle, counter restarted at 0 → center found again.
        self.assertEqual(center_relative(rel_now=0, abs_now=5, center_abs=275), -90)


class TestCenterFromCalibration(unittest.TestCase):
    def write(self, d, data):
        path = os.path.join(d, "w.json")
        with open(path, "w") as f:
            json.dump(data, f)
        return path

    def test_saved_center(self):
        with tempfile.TemporaryDirectory() as d:
            path = self.write(d, {"blind_arcs": [], "info": {"center_abs": 275}, "samples": []})
            self.assertEqual(load_center_abs(path), 275)

    def test_derived_from_first_samples_of_older_file(self):
        # The real file: sweep -90..+90 started at abs 185, 187, ...
        with tempfile.TemporaryDirectory() as d:
            path = self.write(d, {"blind_arcs": [], "info": {"sweep_from": -90},
                                  "samples": [{"abs_deg": 185, "mm": 790}, {"abs_deg": 187, "mm": 800}]})
            self.assertEqual(load_center_abs(path), 275)

    def test_missing_file(self):
        self.assertIsNone(load_center_abs("/nonexistent/whacker_calibration.json"))


class TestManualMove(unittest.TestCase):
    """U4.7 -- RT/LT become hub moves to the end points, so they can't pass them."""

    def setUp(self):
        self.g = ArmGeometry(center=0, base_side=1, base_deg=90, strike_deg=90)

    def test_released_is_stop(self):
        self.assertIsNone(manual_move(0, self.g))

    def test_rt_targets_strike_end_lt_targets_base(self):
        self.assertEqual(manual_move(100, self.g), (-90, 100))
        self.assertEqual(manual_move(-100, self.g), (90, 100))

    def test_follows_base_side(self):
        g = ArmGeometry(center=0, base_side=-1, base_deg=90, strike_deg=90)
        self.assertEqual(manual_move(100, g)[0], 90)
        self.assertEqual(manual_move(-100, g)[0], -90)

    def test_speed_follows_trigger_rounded(self):
        self.assertEqual(manual_move(47, self.g, step=10), (-90, 50))
        self.assertEqual(manual_move(-3, self.g, step=10), (90, 10), "never rounds down to 0")

    def test_small_wobble_gives_same_command(self):
        self.assertEqual(manual_move(51, self.g, step=10), manual_move(54, self.g, step=10))

    def test_holding_a_trigger_never_passes_the_end(self):
        # Simulated hub: overshoot is impossible because it drives TO the target.
        for speed, end in ((100, -90), (-100, 90)):
            arm = FakeArm(start=0, rate=30)
            target, _ = manual_move(speed, self.g)
            positions = []
            for _ in range(100):
                arm.drive_to(target)
                positions.append(arm.pos)
            self.assertTrue(all(-90 <= p <= 90 for p in positions))
            self.assertEqual(arm.pos, end)


class TestArmMotion(unittest.TestCase):
    DT = 1 / 30

    def setUp(self):
        self.g = ArmGeometry(center=0, base_side=1, base_deg=90, strike_deg=90)

    def run_move(self, arm, motion, phase, max_ticks=200):
        t = 0.0
        motion.start(phase, t)
        seen = []
        for _ in range(max_ticks):
            target = motion.step(arm.pos, t)
            if not motion.active:
                break
            arm.drive_to(target)
            seen.append(arm.pos)
            t += self.DT
        return seen, t

    def test_inactive(self):
        m = ArmMotion(self.g)
        self.assertIsNone(m.step(0, 0))
        self.assertFalse(m.active)

    def test_strike_sweeps_base_to_far_end(self):
        arm, m = FakeArm(start=90), ArmMotion(self.g, timeout_s=1.0, tol=5)
        seen, _ = self.run_move(arm, m, STRIKE)
        self.assertAlmostEqual(arm.pos, -90, delta=5)
        self.assertTrue(any(abs(p) <= 20 for p in seen), "passes through center")
        self.assertGreaterEqual(min(seen), -90, "doesn't overshoot the far end")

    def test_home_returns_to_base(self):
        arm, m = FakeArm(start=-90), ArmMotion(self.g, timeout_s=1.0, tol=5)
        self.run_move(arm, m, HOME)
        self.assertAlmostEqual(arm.pos, 90, delta=5)

    def test_targets(self):
        m = ArmMotion(self.g)
        m.start(STRIKE, 0)
        self.assertEqual(m.target(), -90)
        m.start(HOME, 0)
        self.assertEqual(m.target(), 90)

    def test_blocked_strike_times_out(self):
        arm, m = FakeArm(start=90, jam=(30, 100)), ArmMotion(self.g, timeout_s=1.0, tol=5)
        _, t = self.run_move(arm, m, STRIKE)
        self.assertFalse(m.active)
        self.assertLess(t, 1.1)

    def test_cancel(self):
        m = ArmMotion(self.g)
        m.start(STRIKE, 0)
        m.cancel()
        self.assertIsNone(m.step(0, 0.1))


def simulate_sweep(home_abs, blocks, start=-90, end=90, step=3, abs_dir=1, far_mm=None):
    """Simulated hw_whacker_calibrate sweep: base+start → base+end and back.
    `blocks(abs_deg)` says whether the arm is in front of the sensor there
    (sensor then reads 60 mm); elsewhere it reads far_mm (None = nothing seen).
    abs_dir=-1 models an absolute encoder counting the other way."""
    forward = list(range(start, end + 1, step))
    samples = []
    for rel in forward + forward[::-1]:
        a = (home_abs + abs_dir * rel) % 360
        samples.append(SweepSample(abs_deg=a, mm=60 if blocks(a) else far_mm))
    return samples


class TestBlindZones(unittest.TestCase):
    """U4.5 -- whacker blind-zone calibration (simulated 180° sweep)."""

    def test_finds_the_zone_in_a_180_sweep(self):
        samples = simulate_sweep(home_abs=200, blocks=lambda a: 230 <= a <= 260)
        arcs = blind_arcs_from_sweep(samples, detect_mm=300, margin_deg=0)
        self.assertEqual(len(arcs), 2, "one run per sweep direction")
        for arc in arcs:
            self.assertEqual((arc.start, arc.end), (230, 260))

    def test_margin_pads_both_sides(self):
        samples = simulate_sweep(home_abs=200, blocks=lambda a: 230 <= a <= 260)
        arcs = blind_arcs_from_sweep(samples, detect_mm=300, margin_deg=8)
        self.assertTrue(in_blind_zone(223, arcs))
        self.assertTrue(in_blind_zone(267, arcs))
        self.assertFalse(in_blind_zone(220, arcs))
        self.assertFalse(in_blind_zone(270, arcs))

    def test_base_and_edges_stay_clear(self):
        samples = simulate_sweep(home_abs=200, blocks=lambda a: 230 <= a <= 260)
        arcs = blind_arcs_from_sweep(samples, detect_mm=300, margin_deg=8)
        self.assertFalse(in_blind_zone(200, arcs), "base position is clear")
        self.assertFalse(in_blind_zone(290, arcs), "far edge of the swing is clear")
        self.assertFalse(in_blind_zone(110, arcs), "other side is clear")

    def test_zone_crossing_zero(self):
        samples = simulate_sweep(home_abs=330, blocks=lambda a: a >= 350 or a <= 15)
        # 3° sweep steps land on 351°, not 350°: the margin must cover one step.
        arcs = blind_arcs_from_sweep(samples, detect_mm=300, margin_deg=3)
        for a in (350, 359, 0, 10, 15):
            self.assertTrue(in_blind_zone(a, arcs), a)
        for a in (340, 25, 180):
            self.assertFalse(in_blind_zone(a, arcs), a)

    def test_encoder_counting_the_other_way(self):
        samples = simulate_sweep(home_abs=200, blocks=lambda a: 140 <= a <= 170, abs_dir=-1)
        arcs = blind_arcs_from_sweep(samples, detect_mm=300, margin_deg=0)
        self.assertTrue(in_blind_zone(155, arcs))
        self.assertFalse(in_blind_zone(200, arcs))
        self.assertFalse(in_blind_zone(250, arcs), "didn't pick the complementary arc")

    def test_zone_at_the_sweep_turnaround(self):
        # Blind right at the far end: one run goes out, turns around, comes back.
        samples = simulate_sweep(home_abs=200, blocks=lambda a: a >= 270)
        arcs = blind_arcs_from_sweep(samples, detect_mm=300, margin_deg=0)
        self.assertEqual(len(arcs), 1)
        self.assertTrue(in_blind_zone(275, arcs))
        self.assertTrue(in_blind_zone(290, arcs))
        self.assertFalse(in_blind_zone(260, arcs))

    def test_two_separate_zones(self):
        samples = simulate_sweep(home_abs=180, blocks=lambda a: 120 <= a <= 135 or 220 <= a <= 240)
        arcs = blind_arcs_from_sweep(samples, detect_mm=300, margin_deg=0)
        self.assertTrue(in_blind_zone(128, arcs))
        self.assertTrue(in_blind_zone(230, arcs))
        self.assertFalse(in_blind_zone(180, arcs))

    def test_far_readings_are_not_the_whacker(self):
        samples = simulate_sweep(home_abs=200, blocks=lambda a: False, far_mm=900)
        self.assertEqual(blind_arcs_from_sweep(samples, detect_mm=300), [])

    def test_whole_circle_when_always_blocked(self):
        samples = simulate_sweep(home_abs=0, blocks=lambda a: True, start=-180, end=180)
        arcs = blind_arcs_from_sweep(samples, detect_mm=300, margin_deg=8)
        self.assertTrue(all(in_blind_zone(a, arcs) for a in range(0, 360, 10)))

    def test_unknown_whacker_angle_counts_as_blind(self):
        self.assertTrue(in_blind_zone(math.nan, []))
        self.assertTrue(in_blind_zone(None, []))

    def test_visible_distance(self):
        arcs = [Arc(230, 260)]
        self.assertEqual(visible_distance(150, 200, arcs), 150)
        self.assertIsNone(visible_distance(150, 245, arcs))
        self.assertIsNone(visible_distance(150, math.nan, arcs))

    def test_calibration_file_round_trip(self):
        samples = simulate_sweep(home_abs=200, blocks=lambda a: 230 <= a <= 260)
        arcs = blind_arcs_from_sweep(samples, detect_mm=300, margin_deg=8)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "w.json")
            self.assertIsNone(load_blind_calibration(path))
            save_blind_calibration(arcs, samples, path, sweep_from=-90, sweep_to=90)
            self.assertEqual(load_blind_calibration(path), arcs)


if __name__ == "__main__":
    unittest.main()
