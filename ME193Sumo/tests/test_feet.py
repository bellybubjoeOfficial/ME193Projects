"""U3.1 -- base-position arcs + settling speed."""

import os
import tempfile
import unittest

from feet import Arc, arc_from_marks, load_calibration, save_calibration, settle_speed


class TestArc(unittest.TestCase):
    def test_simple_arc(self):
        arc = Arc(80, 120)
        self.assertTrue(arc.contains(80))
        self.assertTrue(arc.contains(100))
        self.assertTrue(arc.contains(120))
        self.assertFalse(arc.contains(79))
        self.assertFalse(arc.contains(121))
        self.assertEqual(arc.width, 40)

    def test_arc_wrapping_past_zero(self):
        arc = Arc(340, 20)
        self.assertEqual(arc.width, 40)
        for a in (340, 359, 0, 10, 20):
            self.assertTrue(arc.contains(a), a)
        for a in (21, 180, 339):
            self.assertFalse(arc.contains(a), a)

    def test_negative_encoder_angles(self):
        # Some hubs report -180..179 -- same physical angle must match.
        arc = Arc(340, 20)
        self.assertTrue(arc.contains(-10))
        self.assertFalse(arc.contains(-90))


class TestArcMiddle(unittest.TestCase):
    def test_center_half(self):
        self.assertEqual(Arc(80, 120).middle(0.5), Arc(90, 110))

    def test_center_half_wrapping(self):
        mid = Arc(340, 20).middle(0.5)
        self.assertEqual((mid.start, mid.end), (350, 10))

    def test_center_half_of_real_left_foot(self):
        # Left foot calibration from the robot: 359° → 182° (183° wide).
        mid = Arc(359, 182).middle(0.5)
        self.assertAlmostEqual(mid.width, 91.5)
        self.assertTrue(mid.contains(90))          # middle of the range
        self.assertFalse(mid.contains(5))          # near an end: base, not target
        self.assertTrue(Arc(359, 182).contains(5))

    def test_full_fraction_is_whole_arc(self):
        self.assertEqual(Arc(80, 120).middle(1.0), Arc(80, 120))


class TestArcFromMarks(unittest.TestCase):
    def test_turned_up_from_first_mark(self):
        self.assertEqual(arc_from_marks(80, 120, travelled=40), Arc(80, 120))

    def test_turned_down_from_first_mark(self):
        self.assertEqual(arc_from_marks(120, 80, travelled=-40), Arc(80, 120))

    def test_left_foot_bug_case_picks_the_longer_side(self):
        # Real left foot: marks 359° and 182°, turned 183° *up* through 0°.
        # The old "shorter arc" rule picked 182 → 359 here (wrong side).
        arc = arc_from_marks(359, 182, travelled=183)
        self.assertEqual(arc, Arc(359, 182))
        self.assertTrue(arc.contains(90))
        self.assertFalse(arc.contains(270))

    def test_same_marks_turned_the_other_way(self):
        arc = arc_from_marks(359, 182, travelled=-177)
        self.assertEqual(arc, Arc(182, 359))
        self.assertTrue(arc.contains(270))

    def test_real_right_foot(self):
        self.assertEqual(arc_from_marks(181, 351, travelled=170), Arc(181, 351))

    def test_negative_marks_normalized(self):
        self.assertEqual(arc_from_marks(-10, 20, travelled=30), Arc(350, 20))


class TestSettleSpeed(unittest.TestCase):
    ARC = Arc(80, 120)

    def test_in_arc_stops(self):
        self.assertEqual(settle_speed(100, self.ARC, 1, 30), 0)
        self.assertEqual(settle_speed(100, self.ARC, -1, 30), 0)

    def test_keeps_last_direction(self):
        self.assertEqual(settle_speed(200, self.ARC, 1, 30), 30)
        self.assertEqual(settle_speed(200, self.ARC, -1, 30), -30)

    def test_never_moved_goes_forward(self):
        self.assertEqual(settle_speed(200, self.ARC, 0, 30), 30)

    def test_reverse_not_allowed_settles_forward(self):
        self.assertEqual(settle_speed(200, self.ARC, -1, 30, allow_reverse=False), 30)


class TestCalibrationFile(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "cal.json")
            self.assertIsNone(load_calibration(path))
            save_calibration(Arc(80, 120), Arc(340, 20), path)
            self.assertEqual(load_calibration(path), (Arc(80, 120), Arc(340, 20)))


if __name__ == "__main__":
    unittest.main()
