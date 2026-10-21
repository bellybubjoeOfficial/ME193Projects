"""U5.1 -- sound buttons (Y = sad, A = victory) + tune synthesis."""

import unittest

import numpy as np

from policy import Inputs, decide
from sounds import SAD_NOTES, VICTORY_NOTES, melody_wave
from xbox import ButtonEdges, XboxState


def pad(**kw):
    return XboxState(connected=True, **kw)


class TestButtonEdges(unittest.TestCase):
    def test_fires_once_per_press(self):
        e = ButtonEdges()
        self.assertEqual(e.update(pad(y=True)), {"y"})
        self.assertEqual(e.update(pad(y=True)), set(), "holding doesn't repeat")
        self.assertEqual(e.update(pad()), set())
        self.assertEqual(e.update(pad(y=True)), {"y"}, "press again → fires again")

    def test_independent_buttons(self):
        e = ButtonEdges()
        e.update(pad(y=True))
        self.assertEqual(e.update(pad(y=True, a=True)), {"a"})

    def test_disconnected_releases_everything(self):
        e = ButtonEdges()
        e.update(pad(a=True))
        self.assertEqual(e.update(XboxState(connected=False, a=True)), set())
        self.assertEqual(e.update(pad(a=True)), {"a"}, "reconnect with A held counts as a press")


class TestSoundBranch(unittest.TestCase):
    def test_y_is_sad_a_is_victory(self):
        self.assertEqual(decide(Inputs(pad=pad(y=True), pressed=frozenset({"y"}))).sound, "sad")
        self.assertEqual(decide(Inputs(pad=pad(a=True), pressed=frozenset({"a"}))).sound, "victory")

    def test_held_button_without_new_press_is_silent(self):
        self.assertIsNone(decide(Inputs(pad=pad(y=True))).sound)

    def test_other_buttons_silent(self):
        self.assertIsNone(decide(Inputs(pad=pad(x=True), pressed=frozenset({"x"}))).sound)

    def test_sound_doesnt_change_driving_or_whacker(self):
        a = decide(Inputs(pad=pad(ly=1.0, rt=1.0, a=True), pressed=frozenset({"a"})))
        self.assertEqual(a.sound, "victory")
        self.assertGreater(a.left, 0)
        self.assertEqual(a.whack_mode, "manual")

    def test_sound_still_plays_while_braking(self):
        a = decide(Inputs(pad=pad(b=True, y=True), pressed=frozenset({"y"})))
        self.assertEqual((a.reason, a.sound), ("brake", "sad"))


class TestMelody(unittest.TestCase):
    RATE = 44100

    def test_length_matches_notes(self):
        for notes in (SAD_NOTES, VICTORY_NOTES):
            wave = melody_wave(notes, self.RATE, brass=True)
            expected = sum(int(self.RATE * d) for _, d in notes)
            self.assertEqual(len(wave), expected)

    def test_never_clips(self):
        for notes in (SAD_NOTES, VICTORY_NOTES):
            wave = melody_wave(notes, self.RATE, brass=True, vibrato_last=True)
            self.assertLessEqual(np.max(np.abs(wave)), 1.0)

    def test_sad_goes_down_victory_goes_up(self):
        sad = [f for f, _ in SAD_NOTES if f]
        win = [f for f, _ in VICTORY_NOTES if f]
        self.assertEqual(sad, sorted(sad, reverse=True))
        self.assertLess(win[0], win[-1])

    def test_starts_and_ends_silent(self):
        wave = melody_wave(SAD_NOTES, self.RATE)
        self.assertAlmostEqual(wave[0], 0.0, places=3)
        self.assertAlmostEqual(wave[-1], 0.0, places=2)


if __name__ == "__main__":
    unittest.main()
