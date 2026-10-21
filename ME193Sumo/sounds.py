"""
sounds.py

Synthesized sound effects (no audio files), same tone-synthesis idea as
ME193Midterm/sound.py.

  edge     two quick beeps          (color sensor hits the tape)
  sad      "wah wah wah wahhh"      (Y button; also the ring-out song later)
  victory  rising fanfare           (A button)
"""

import numpy as np
import pygame

# (frequency Hz, seconds); frequency 0 = rest
SAD_NOTES = [(392.0, 0.45), (370.0, 0.45), (349.2, 0.45), (329.6, 1.3)]       # G4 F#4 F4 E4
VICTORY_NOTES = [(523.3, 0.13), (659.3, 0.13), (784.0, 0.13), (1046.5, 0.28),
                 (0, 0.06), (784.0, 0.13), (1046.5, 0.6)]                       # C5 E5 G5 C6 . G5 C6


def _mixer_format():
    rate, _, channels = pygame.mixer.get_init()
    return rate, channels


def _to_sound(wave, volume):
    """Float wave (-1..1) → pygame Sound in the mixer's own format
    (pygame.init() may already have opened the mixer)."""
    _, channels = _mixer_format()
    samples = (np.clip(wave, -1, 1) * volume * 32767).astype(np.int16)
    if channels > 1:
        samples = np.repeat(samples[:, None], channels, axis=1)
    return pygame.sndarray.make_sound(np.ascontiguousarray(samples))


def _tone(freq_hz, duration_s, decay=6.0, volume=0.6):
    """Sine tone with exponential decay: a percussive "ping"."""
    rate, _ = _mixer_format()
    t = np.arange(int(rate * duration_s)) / rate
    return _to_sound(np.sin(2 * np.pi * freq_hz * t) * np.exp(-t * decay), volume)


def melody_wave(notes, rate, brass=False, vibrato_last=False):
    """Notes → float wave. Each note gets a short attack/release so they
    don't click. brass=True adds harmonics (trombone-ish); vibrato_last
    wobbles the final note (the "wahhh")."""
    parts = []
    for i, (freq, dur) in enumerate(notes):
        n = int(rate * dur)
        t = np.arange(n) / rate
        if freq == 0:
            parts.append(np.zeros(n))
            continue
        f = freq * (1 + 0.015 * np.sin(2 * np.pi * 5.5 * t)) if vibrato_last and i == len(notes) - 1 else freq
        phase = 2 * np.pi * np.cumsum(np.broadcast_to(f, t.shape)) / rate
        wave = np.sin(phase)
        if brass:
            wave = wave + 0.5 * np.sin(2 * phase) + 0.25 * np.sin(3 * phase)
            wave /= 1.75
        env = np.ones(n)
        a, r = min(n // 4, int(0.02 * rate)), min(n // 3, int(0.08 * rate))
        env[:a] = np.linspace(0, 1, a)
        env[n - r:] = np.linspace(1, 0, r)
        parts.append(wave * env)
    return np.concatenate(parts)


class SoundEffects:
    def __init__(self):
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=44100, size=-16, channels=2)
        rate, _ = _mixer_format()
        # Two quick high beeps: "edge!"
        beep = _tone(1500, 0.12, decay=4.0)
        arr = pygame.sndarray.array(beep)
        silence = np.zeros((int(rate * 0.05),) + arr.shape[1:], dtype=arr.dtype)
        self._edge = pygame.sndarray.make_sound(np.ascontiguousarray(np.concatenate([arr, silence, arr])))
        self._sad = _to_sound(melody_wave(SAD_NOTES, rate, brass=True, vibrato_last=True), 0.6)
        self._victory = _to_sound(melody_wave(VICTORY_NOTES, rate, brass=True), 0.55)
        self._tune_channel = None

    def play_edge(self):
        self._edge.play()

    def _play_tune(self, sound):
        # Only one tune at a time: a new one cuts off the previous.
        if self._tune_channel is not None:
            self._tune_channel.stop()
        self._tune_channel = sound.play()

    def play_sad(self):
        self._play_tune(self._sad)

    def play_victory(self):
        self._play_tune(self._victory)

    def play(self, name):
        """Play by name: "edge", "sad" or "victory"."""
        {"edge": self.play_edge, "sad": self.play_sad, "victory": self.play_victory}[name]()
