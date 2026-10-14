"""
sound.py

Short synthesized sound effects (hit/score/miss) -- no external asset
files, generated procedurally (same tone-synthesis idea as
ME_193_HW3/songs.py, just through pygame's own audio mixer instead of raw
pyaudio, since pygame already owns the audio device here).
"""

import numpy as np
import pygame

SAMPLE_RATE = 44100


def _tone(freq_hz, duration_s, decay=18.0, volume=0.5):
    """A sine tone with a quick exponential decay, so it sounds like a
    percussive "ping" rather than a sustained note."""
    n_samples = int(SAMPLE_RATE * duration_s)
    t = np.arange(n_samples) / SAMPLE_RATE
    wave = np.sin(2 * np.pi * freq_hz * t)
    envelope = np.exp(-t * decay)
    samples = (wave * envelope * volume * 32767).astype(np.int16)
    stereo = np.ascontiguousarray(np.column_stack([samples, samples]))
    return pygame.sndarray.make_sound(stereo)


class SoundEffects:
    def __init__(self):
        pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2)
        self._hit = _tone(1200, 0.12)
        self._score = _tone(1800, 0.25, decay=10.0)
        self._miss = _tone(220, 0.35, decay=6.0)

    def play_hit(self):
        self._hit.play()

    def play_score(self):
        self._score.play()

    def play_miss(self):
        self._miss.play()
