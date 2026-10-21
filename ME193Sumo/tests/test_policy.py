"""U1.3 / U2.2 / U2.3 / U3.3 / U3.4 / U4.3 / U4.4 -- the decision tree."""

import math
import unittest

import config
from feet import Arc, FeetState, sign
from policy import Inputs, decide
from ring import REVERSE, EdgeGuard
from whacker import HOME, STRIKE, ArmGeometry, ArmMotion, ProximityTrigger, manual_move, visible_distance
from xbox import XboxState


def pad(**kw):
    return XboxState(connected=True, **kw)


class TestTreeV1(unittest.TestCase):
    def test_controller_lost_stops_even_with_stick_pushed(self):
        a = decide(Inputs(pad=XboxState(connected=False, ly=1.0)))
        self.assertEqual((a.left, a.right, a.reason), (0, 0, "controller lost"))

    def test_brake_beats_stick(self):
        a = decide(Inputs(pad=pad(b=True, ly=1.0)))
        self.assertEqual((a.left, a.right, a.reason), (0, 0, "brake"))

    def test_stick_forward_drives_forward(self):
        a = decide(Inputs(pad=pad(ly=1.0)))
        self.assertEqual(a.reason, "teleop drive")
        self.assertGreater(a.left, 0)
        self.assertEqual(a.left, a.right)

    def test_stick_right_turns_right(self):
        a = decide(Inputs(pad=pad(lx=1.0)))
        self.assertGreater(a.left, a.right)

    def test_released_stick_is_zero(self):
        a = decide(Inputs(pad=pad()))
        self.assertEqual((a.left, a.right), (0, 0))


class TestTreeV2(unittest.TestCase):
    """U2.2 -- edge branch, one tick at a time."""

    def test_tape_hit_raises_alert(self):
        self.assertTrue(decide(Inputs(pad=pad(ly=1.0), tape_hit=True)).alert)
        self.assertFalse(decide(Inputs(pad=pad(ly=1.0))).alert)

    def test_alert_still_fires_when_braking_or_disconnected(self):
        self.assertTrue(decide(Inputs(pad=pad(b=True), tape_hit=True)).alert)
        self.assertTrue(decide(Inputs(pad=XboxState(), tape_hit=True)).alert)

    def test_reverse_blocked_stops_reverse(self):
        a = decide(Inputs(pad=pad(ly=-1.0), blocked_dir=REVERSE))
        self.assertEqual((a.left, a.right), (0, 0))
        self.assertEqual(a.reason, "edge: reverse blocked")

    def test_reverse_blocked_still_allows_forward(self):
        a = decide(Inputs(pad=pad(ly=1.0), blocked_dir=REVERSE))
        self.assertEqual(a.reason, "teleop drive")
        self.assertGreater(a.left, 0)
        self.assertGreater(a.right, 0)

    def test_reverse_blocked_still_allows_forward_arcs(self):
        a = decide(Inputs(pad=pad(ly=1.0, lx=-0.5), blocked_dir=REVERSE))
        self.assertEqual(a.reason, "teleop drive")
        self.assertGreater(a.right, a.left)

    def test_reverse_blocked_still_allows_turning_in_place(self):
        a = decide(Inputs(pad=pad(ly=-1.0, lx=1.0), blocked_dir=REVERSE))
        self.assertEqual(a.left, -a.right)
        self.assertGreater(a.left, 0)


class TestEdgeScenario(unittest.TestCase):
    """U2.3 -- EdgeGuard + decide() run together over simulated time,
    exactly as main.py wires them (30 Hz ticks)."""

    DT = 1 / 30

    def run_ticks(self, guard, stick_ly, on_tape, t0, n):
        actions = []
        t = t0
        for _ in range(n):
            hit = guard.update(on_tape, t)
            a = decide(Inputs(pad=pad(ly=stick_ly), tape_hit=hit,
                              blocked_dir=guard.blocked_direction(t)))
            actions.append((t, a))
            t += self.DT
        return actions, t

    def test_backing_onto_tape(self):
        g = EdgeGuard(block_s=1.5)

        # Backing up inside the ring.
        acts, t = self.run_ticks(g, -1.0, False, 0.0, 10)
        self.assertTrue(all(a.left < 0 for _, a in acts))

        # Sensor hits the tape while still holding reverse.
        acts, t = self.run_ticks(g, -1.0, True, t, 3)
        self.assertTrue(acts[0][1].alert, "alert on the first tape tick")
        self.assertFalse(any(a.alert for _, a in acts[1:]), "alert only once")
        self.assertTrue(all((a.left, a.right) == (0, 0) for _, a in acts),
                        "reverse stops immediately")
        last_blue = acts[-1][0]

        # Driver pushes forward off the tape → allowed right away.
        acts, t = self.run_ticks(g, 1.0, False, t, 3)
        self.assertTrue(all(a.left > 0 and a.right > 0 for _, a in acts))

        # Reverse again inside 1.5 s of the last blue reading → blocked.
        acts, t = self.run_ticks(g, -1.0, False, t, 30)
        self.assertLess(t, last_blue + 1.5)
        self.assertTrue(all((a.left, a.right) == (0, 0) for _, a in acts))

        # After 1.5 s → reverse works again.
        _, t = self.run_ticks(g, -1.0, False, t, int(0.5 / self.DT))
        acts, t = self.run_ticks(g, -1.0, False, t, 1)
        self.assertGreater(acts[0][0], last_blue + 1.5)
        self.assertLess(acts[0][1].left, 0)

    def test_driving_forward_onto_tape_still_only_blocks_reverse(self):
        g = EdgeGuard(block_s=1.5)
        _, t = self.run_ticks(g, 1.0, False, 0.0, 5)
        acts, t = self.run_ticks(g, 1.0, True, t, 3)
        self.assertTrue(acts[0][1].alert)
        self.assertTrue(all(a.left > 0 and a.right > 0 for _, a in acts),
                        "forward keeps working on the tape")
        acts, _ = self.run_ticks(g, -1.0, True, t, 1)
        self.assertEqual((acts[0][1].left, acts[0][1].right), (0, 0))



BASE = Arc(80, 120)


def feet(lp, rp, ld=0, rd=0):
    return FeetState(left_pos=lp, right_pos=rp, left_arc=BASE, right_arc=BASE,
                     left_dir=ld, right_dir=rd)


class TestTreeV3Feet(unittest.TestCase):
    """U3.3 -- settling feet, one tick at a time."""

    S = config.SETTLE_SPEED

    def test_stick_input_ignores_settling(self):
        a = decide(Inputs(pad=pad(ly=1.0), feet=feet(200, 200, -1, -1)))
        self.assertEqual(a.reason, "teleop drive")
        self.assertGreater(a.left, 0)

    def test_idle_with_feet_in_center_stops(self):
        # BASE 80..120 → center 50% target is 90..110.
        a = decide(Inputs(pad=pad(), feet=feet(100, 90, 1, 1)))
        self.assertEqual((a.left, a.right, a.reason), (0, 0, "idle"))

    def test_in_base_but_outside_center_keeps_settling(self):
        a = decide(Inputs(pad=pad(), feet=feet(82, 118, 1, -1)))
        self.assertEqual((a.left, a.right, a.reason), (self.S, -self.S, "settling feet"))

    def test_idle_out_of_base_keeps_last_direction(self):
        a = decide(Inputs(pad=pad(), feet=feet(200, 200, 1, -1)))
        self.assertEqual((a.left, a.right, a.reason), (self.S, -self.S, "settling feet"))

    def test_each_foot_settles_independently(self):
        a = decide(Inputs(pad=pad(), feet=feet(100, 200, 1, 1)))
        self.assertEqual((a.left, a.right), (0, self.S))

    def test_brake_does_not_settle(self):
        a = decide(Inputs(pad=pad(b=True), feet=feet(200, 200, 1, 1)))
        self.assertEqual((a.left, a.right, a.reason), (0, 0, "brake"))

    def test_controller_lost_does_not_settle(self):
        a = decide(Inputs(pad=XboxState(), feet=feet(200, 200, 1, 1)))
        self.assertEqual((a.left, a.right), (0, 0))

    def test_uncalibrated_feet_just_stop(self):
        a = decide(Inputs(pad=pad(), feet=None))
        self.assertEqual((a.left, a.right), (0, 0))

    def test_reverse_blocked_settles_forward(self):
        a = decide(Inputs(pad=pad(), blocked_dir=REVERSE, feet=feet(200, 200, -1, -1)))
        self.assertEqual((a.left, a.right), (self.S, self.S))

    def test_reverse_blocked_with_stick_back_settles_forward(self):
        a = decide(Inputs(pad=pad(ly=-1.0), blocked_dir=REVERSE, feet=feet(200, 200, -1, -1)))
        self.assertEqual((a.left, a.right), (self.S, self.S))


class TestTreeV3Whacker(unittest.TestCase):
    """U3.4 -- whacker rides alongside every connected branch."""

    def test_rt_and_lt(self):
        self.assertGreater(decide(Inputs(pad=pad(rt=1.0))).whack, 0)
        self.assertLess(decide(Inputs(pad=pad(lt=1.0))).whack, 0)
        self.assertEqual(decide(Inputs(pad=pad())).whack, 0)

    def test_whacker_works_while_driving_braking_or_settling(self):
        self.assertGreater(decide(Inputs(pad=pad(rt=1.0, ly=1.0))).whack, 0)
        self.assertGreater(decide(Inputs(pad=pad(rt=1.0, b=True))).whack, 0)
        self.assertGreater(decide(Inputs(pad=pad(rt=1.0), feet=feet(200, 200, 1, 1))).whack, 0)

    def test_controller_lost_stops_whacker(self):
        self.assertEqual(decide(Inputs(pad=XboxState(rt=1.0))).whack, 0)


class TestSettleScenario(unittest.TestCase):
    """U3.5 -- simulated crank feet + decide(), wired like main.py.
    Each foot turns DEG_PER_UNIT degrees per tick per unit of speed."""

    DEG_PER_UNIT = 0.1   # speed 30 → 3°/tick; speed 100 → 10°/tick

    def test_release_mid_stride_rolls_on_into_base(self):
        lp, rp = 100.0, 100.0          # start in base
        ld = rd = 0
        log = []

        def tick(stick_ly):
            nonlocal lp, rp, ld, rd
            a = decide(Inputs(pad=pad(ly=stick_ly), feet=feet(lp, rp, ld, rd)))
            ld = sign(a.left) or ld
            rd = sign(a.right) or rd
            lp = (lp + a.left * self.DEG_PER_UNIT) % 360
            rp = (rp + a.right * self.DEG_PER_UNIT) % 360
            log.append(a)
            return a

        # Walk backwards for 10 ticks → feet now out of base, last dir = reverse.
        for _ in range(10):
            tick(-1.0)
        self.assertFalse(BASE.contains(lp))

        # Let go: feet must keep turning in REVERSE (same direction) until in base.
        for _ in range(400):
            a = tick(0.0)
            if a.reason == "idle":
                break
            self.assertEqual(a.reason, "settling feet")
            self.assertLessEqual(a.left, 0, "never flips direction")
            self.assertLessEqual(a.right, 0)
        self.assertEqual(log[-1].reason, "idle")
        target = BASE.middle(config.SETTLE_TARGET_FRACTION)
        self.assertTrue(target.contains(lp) and target.contains(rp),
                        "feet stop in the center 50%, not just anywhere in base")

        # And it stays put once settled.
        for _ in range(5):
            self.assertEqual(tick(0.0).reason, "idle")



class TestTreeV5Whacker(unittest.TestCase):
    """U4.3 -- whacker sub-tree (base / strike / return), one tick at a time."""

    HOLD = config.WHACKER_MANUAL_HOLD_S

    def test_opponent_at_base_starts_strike(self):
        a = decide(Inputs(pad=pad(), opponent_entered=True, arm_at_base=True))
        self.assertEqual((a.arm_command, a.whack_mode), (STRIKE, "strike start"))

    def test_strike_in_progress_continues(self):
        a = decide(Inputs(pad=pad(), arm_phase=STRIKE, arm_target=-90, arm_at_base=False))
        self.assertEqual((a.whack_mode, a.whack_target, a.arm_command), ("strike", -90, None))

    def test_no_restart_mid_strike(self):
        a = decide(Inputs(pad=pad(), opponent_entered=True, arm_phase=STRIKE, arm_target=-90))
        self.assertIsNone(a.arm_command)

    def test_opponent_while_returning_strikes_again(self):
        a = decide(Inputs(pad=pad(), opponent_entered=True, arm_phase=HOME, arm_target=90))
        self.assertEqual(a.arm_command, STRIKE)

    def test_returning_continues(self):
        a = decide(Inputs(pad=pad(), arm_phase=HOME, arm_target=90, arm_at_base=False))
        self.assertEqual((a.whack_mode, a.whack_target), ("home", 90))

    def test_away_from_base_and_idle_goes_home(self):
        a = decide(Inputs(pad=pad(), arm_at_base=False, manual_idle_s=self.HOLD + 0.1))
        self.assertEqual((a.arm_command, a.whack_mode), (HOME, "home start"))

    def test_startup_never_touched_goes_home(self):
        a = decide(Inputs(pad=pad(), arm_at_base=False))           # manual_idle_s = inf
        self.assertEqual(a.arm_command, HOME)

    def test_recent_manual_holds_position(self):
        a = decide(Inputs(pad=pad(), arm_at_base=False, manual_idle_s=self.HOLD - 0.1))
        self.assertEqual((a.arm_command, a.whack_mode), (None, "off"))

    def test_at_base_holds(self):
        a = decide(Inputs(pad=pad(), arm_at_base=True))
        self.assertEqual((a.arm_command, a.whack_mode, a.whack), (None, "off", 0))

    def test_unknown_position_does_nothing(self):
        a = decide(Inputs(pad=pad(), arm_at_base=None))
        self.assertIsNone(a.arm_command)

    def test_manual_overrides_and_cancels_any_move(self):
        for phase in (STRIKE, HOME):
            a = decide(Inputs(pad=pad(rt=1.0), arm_phase=phase, arm_target=0))
            self.assertEqual((a.whack_mode, a.arm_command), ("manual", "cancel"))
            self.assertGreater(a.whack, 0)

    def test_manual_blocks_new_strike(self):
        a = decide(Inputs(pad=pad(lt=0.5), opponent_entered=True, arm_at_base=True))
        self.assertEqual(a.whack_mode, "manual")
        self.assertIsNone(a.arm_command)

    def test_controller_lost_cancels(self):
        a = decide(Inputs(pad=XboxState(), arm_phase=STRIKE, arm_target=-90))
        self.assertEqual((a.whack, a.arm_command), (0, "cancel"))

    def test_strike_while_driving_and_braking(self):
        self.assertEqual(decide(Inputs(pad=pad(ly=1.0), opponent_entered=True)).arm_command, STRIKE)
        self.assertEqual(decide(Inputs(pad=pad(b=True), opponent_entered=True)).arm_command, STRIKE)


class TestWhackerScenario(unittest.TestCase):
    """U4.4 -- distance → ProximityTrigger → decide() → ArmMotion, wired like
    main.py, with a simulated hub-controlled arm and simulated time."""

    DT = 1 / 30

    def setUp(self):
        self.g = ArmGeometry(center=0, base_side=1, base_deg=90, strike_deg=90)
        self.motion = ArmMotion(self.g, timeout_s=1.0, tol=5)
        self.prox = ProximityTrigger(trigger_mm=200, rearm_mm=250)
        self.pos = 0.0                       # program starts with the arm at center
        self.t = 0.0
        self.last_manual = -math.inf
        self.strikes = 0
        self.log = []

    def tick(self, mm=None, **pad_kw):
        entered = self.prox.update(mm)
        target = self.motion.step(self.pos, self.t) if self.motion.active else None
        a = decide(Inputs(pad=pad(**pad_kw), opponent_entered=entered,
                          arm_phase=self.motion.phase, arm_target=target,
                          arm_at_base=self.g.at_base(self.pos),
                          manual_idle_s=self.t - self.last_manual))
        if a.arm_command == "cancel":
            self.motion.cancel()
        elif a.arm_command in (STRIKE, HOME):
            self.motion.start(a.arm_command, self.t)
            self.strikes += a.arm_command == STRIKE
        if a.whack_mode == "manual":
            self.last_manual = self.t
        # simulated arm, commanded like main.py: every move is a hub position move
        move = None
        if a.whack_mode in (STRIKE, HOME):
            move = (a.whack_target, config.WHACKER_SPEED)
        elif a.whack_mode == "manual":
            move = manual_move(a.whack, self.g)
        if move:
            self.pos += max(-20.0, min(20.0, move[0] - self.pos))
        self.max_seen = max(getattr(self, "max_seen", self.pos), self.pos)
        self.min_seen = min(getattr(self, "min_seen", self.pos), self.pos)
        self.t += self.DT
        self.log.append((self.t, a, self.pos))
        return a

    def ticks(self, n, mm=None, **pad_kw):
        for _ in range(n):
            self.tick(mm, **pad_kw)

    def test_startup_goes_from_center_to_base(self):
        self.ticks(30, mm=900)
        self.assertAlmostEqual(self.pos, 90, delta=5)
        self.assertEqual(self.log[-1][1].whack_mode, "off", "holding at base")

    def test_full_strike_and_automatic_return(self):
        self.ticks(30, mm=900)                         # settle at base
        self.ticks(30, mm=150)                         # opponent close for 1 s
        positions = [p for _, _, p in self.log[30:]]
        self.assertEqual(self.strikes, 1, "exactly one strike while they sit close")
        self.assertLessEqual(min(positions), -85, "swept all the way to the far end")
        self.assertAlmostEqual(self.pos, 90, delta=5, msg="came straight back to base")

        self.ticks(5, mm=300)                          # back off past 250 → re-armed
        self.ticks(30, mm=150)
        self.assertEqual(self.strikes, 2)

    def test_manual_then_two_second_hold_then_home(self):
        self.ticks(30, mm=900)                         # at base
        self.ticks(3, mm=900, rt=0.5)                # driver swings it partway (RT = toward strike end)
        moved_to = self.pos
        self.assertLess(moved_to, 80)
        released_at = self.t
        # Within 2 s of letting go: stays where the driver left it.
        while self.t < released_at + config.WHACKER_MANUAL_HOLD_S - 0.1:
            self.tick(900)
        self.assertAlmostEqual(self.pos, moved_to, delta=0.01)
        # After 2 s: returns to base.
        self.ticks(30, mm=900)
        self.assertAlmostEqual(self.pos, 90, delta=5)

    def test_driver_grabs_mid_strike_and_it_waits_before_returning(self):
        self.ticks(30, mm=900)
        self.tick(150)                               # strike starts
        self.ticks(3, mm=150)
        self.ticks(5, mm=150, rt=1.0)                  # driver takes over mid-strike
        self.assertFalse(self.motion.active)
        grabbed = self.pos
        self.ticks(int(1.5 / self.DT), mm=150)         # 1.5 s idle: still held
        self.assertAlmostEqual(self.pos, grabbed, delta=0.01)
        self.assertEqual(self.strikes, 1, "no new strike while they're still close")
        self.ticks(30, mm=150)                         # past 2 s → home
        self.assertAlmostEqual(self.pos, 90, delta=5)

    def test_unknown_distance_never_strikes(self):
        self.ticks(60, mm=None)
        self.assertEqual(self.strikes, 0)

    def test_xbox_can_never_pass_the_end_points(self):
        self.ticks(30, mm=900)                       # at base (+90)
        self.ticks(90, mm=900, rt=1.0)               # hold RT for 3 s
        self.assertAlmostEqual(self.pos, -90, delta=0.01, msg="stops exactly at the strike end")
        self.ticks(90, mm=900, lt=1.0)               # hold LT for 3 s
        self.assertAlmostEqual(self.pos, 90, delta=0.01, msg="stops exactly at base")
        self.assertGreaterEqual(self.min_seen, -90)
        self.assertLessEqual(self.max_seen, 90)


class TestBlindZoneScenario(unittest.TestCase):
    """U4.6 -- the whacker in front of the sensor never triggers itself."""

    BLIND = [Arc(230, 260)]

    def test_whacker_in_front_of_sensor_is_ignored(self):
        prox = ProximityTrigger(trigger_mm=200, rearm_mm=250)
        # Arm sweeps through the blind zone; the sensor "sees" it at 60 mm.
        for abs_deg in range(200, 290, 5):
            raw = 60 if 230 <= abs_deg <= 260 else 900
            self.assertFalse(prox.update(visible_distance(raw, abs_deg, self.BLIND)), abs_deg)

    def test_real_opponent_still_triggers_when_arm_is_clear(self):
        prox = ProximityTrigger(trigger_mm=200, rearm_mm=250)
        self.assertTrue(prox.update(visible_distance(150, 200, self.BLIND)))

    def test_blind_zone_during_a_whack_does_not_rearm(self):
        prox = ProximityTrigger(trigger_mm=200, rearm_mm=250)
        prox.update(visible_distance(150, 200, self.BLIND))      # whack #1
        prox.update(visible_distance(900, 245, self.BLIND))      # far reading but arm in front → ignored
        self.assertFalse(prox.update(visible_distance(150, 200, self.BLIND)),
                         "a blind reading must not count as the opponent backing off")


if __name__ == "__main__":
    unittest.main()
