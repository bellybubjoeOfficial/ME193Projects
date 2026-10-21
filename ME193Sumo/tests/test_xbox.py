"""U1.1 -- raw SDL controller values → XboxState."""

import unittest

from xbox import AXIS_MAX, apply_deadzone, state_from_raw


class TestDeadzone(unittest.TestCase):
    def test_inside_deadzone_is_zero(self):
        self.assertEqual(apply_deadzone(0.05, 0.12), 0.0)
        self.assertEqual(apply_deadzone(-0.11, 0.12), 0.0)

    def test_full_travel_is_one(self):
        self.assertAlmostEqual(apply_deadzone(1.0, 0.12), 1.0)
        self.assertAlmostEqual(apply_deadzone(-1.0, 0.12), -1.0)

    def test_ramps_from_zero_at_edge(self):
        self.assertAlmostEqual(apply_deadzone(0.12, 0.12), 0.0)
        self.assertAlmostEqual(apply_deadzone(0.56, 0.12), 0.5)


class TestStateFromRaw(unittest.TestCase):
    def test_centered_pad_is_all_zero(self):
        s = state_from_raw({"lx": 300, "ly": -500, "rx": 0, "ry": 0, "lt": 0, "rt": 0}, {})
        self.assertTrue(s.connected)
        self.assertEqual((s.lx, s.ly, s.rx, s.ry, s.lt, s.rt), (0, 0, 0, 0, 0, 0))

    def test_stick_up_is_positive(self):
        # SDL reports stick-up as negative Y; we flip it.
        s = state_from_raw({"ly": -int(AXIS_MAX)}, {})
        self.assertAlmostEqual(s.ly, 1.0)

    def test_stick_right_is_positive(self):
        s = state_from_raw({"lx": int(AXIS_MAX)}, {})
        self.assertAlmostEqual(s.lx, 1.0)

    def test_min_raw_value_clamps(self):
        s = state_from_raw({"lx": -32768}, {})
        self.assertAlmostEqual(s.lx, -1.0)

    def test_triggers_0_to_1(self):
        s = state_from_raw({"rt": int(AXIS_MAX), "lt": 0}, {})
        self.assertAlmostEqual(s.rt, 1.0)
        self.assertEqual(s.lt, 0.0)

    def test_buttons(self):
        s = state_from_raw({}, {"b": 1, "lb": True})
        self.assertTrue(s.b)
        self.assertTrue(s.lb)
        self.assertFalse(s.a)


if __name__ == "__main__":
    unittest.main()
