"""U1.2 -- arcade mixing of stick → wheel speeds."""

import unittest

from drive import arcade_mix


class TestArcadeMix(unittest.TestCase):
    def mix(self, throttle, turn):
        return arcade_mix(throttle, turn, max_speed=100, turn_gain=1.0)

    def test_still(self):
        self.assertEqual(self.mix(0, 0), (0, 0))

    def test_full_forward(self):
        self.assertEqual(self.mix(1, 0), (100, 100))

    def test_full_reverse(self):
        self.assertEqual(self.mix(-1, 0), (-100, -100))

    def test_spin_right_in_place(self):
        self.assertEqual(self.mix(0, 1), (100, -100))

    def test_spin_left_in_place(self):
        self.assertEqual(self.mix(0, -1), (-100, 100))

    def test_never_exceeds_max(self):
        for t in (-1, -0.5, 0, 0.5, 1):
            for r in (-1, -0.5, 0, 0.5, 1):
                left, right = self.mix(t, r)
                self.assertLessEqual(abs(left), 100)
                self.assertLessEqual(abs(right), 100)

    def test_forward_right_arc_keeps_ratio(self):
        self.assertEqual(self.mix(1, 1), (100, 0))
        # (1.5, 0.5) overshoots, so both sides scale down by 1.5 → (100, 33)
        self.assertEqual(self.mix(1, 0.5), (100, 33))

    def test_turn_gain_softens_turning(self):
        self.assertEqual(arcade_mix(0, 1, max_speed=100, turn_gain=0.5), (50, -50))


if __name__ == "__main__":
    unittest.main()
