"""U2.1 -- tape detection + EdgeGuard reverse blocking."""

import unittest

from ring import NONE, REVERSE, EdgeGuard, is_tape


class TestIsTape(unittest.TestCase):
    def test_blue_is_tape(self):
        self.assertTrue(is_tape("Blue"))

    def test_other_colors_are_not(self):
        for name in ("No color", "White", "Red", "Unknown"):
            self.assertFalse(is_tape(name), name)


class TestEdgeGuard(unittest.TestCase):
    def setUp(self):
        self.g = EdgeGuard(block_s=1.5)

    def test_nothing_blocked_before_any_tape(self):
        self.assertFalse(self.g.update(False, 0.0))
        self.assertEqual(self.g.blocked_direction(0.0), NONE)

    def test_hit_fires_only_on_first_tick(self):
        self.assertTrue(self.g.update(True, 0.0))
        self.assertFalse(self.g.update(True, 0.03))
        self.assertFalse(self.g.update(True, 0.06))

    def test_blue_blocks_reverse_for_1_5_s(self):
        self.g.update(True, 10.0)
        self.g.update(False, 10.03)
        self.assertEqual(self.g.blocked_direction(10.0), REVERSE)
        self.assertEqual(self.g.blocked_direction(11.49), REVERSE)
        self.assertEqual(self.g.blocked_direction(11.5), NONE)

    def test_staying_on_tape_keeps_reverse_blocked(self):
        # 3 s parked on the tape → still blocked, then 1.5 s after leaving.
        t = 0.0
        while t < 3.0:
            self.g.update(True, t)
            t += 0.1
        last_blue = t - 0.1
        self.g.update(False, t)
        self.assertEqual(self.g.blocked_direction(last_blue + 1.4), REVERSE)
        self.assertEqual(self.g.blocked_direction(last_blue + 1.5), NONE)

    def test_relanding_fires_again(self):
        self.g.update(True, 0.0)
        self.g.update(False, 0.3)
        self.assertTrue(self.g.update(True, 0.4))
        self.assertEqual(self.g.blocked_direction(1.8), REVERSE)
        self.assertEqual(self.g.blocked_direction(1.9), NONE)


if __name__ == "__main__":
    unittest.main()
