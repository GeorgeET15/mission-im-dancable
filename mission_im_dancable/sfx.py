"""Tiny synthesized UI sound effects — generated in-code, no audio assets.

Same idea as lobby.py: numpy → int16 stereo → pygame.mixer.Sound played on
whatever channel is free. One-shots don't need channel management.
"""

from __future__ import annotations

import numpy as np

_SR = 44100


def click_samples(sr: int = _SR) -> np.ndarray:
    """Short bright tap (~70 ms) — a quick downward chirp."""
    dur = 0.07
    t = np.arange(int(dur * sr)) / sr
    f0, f1 = 2200.0, 800.0
    phase = 2 * np.pi * (f0 * t + (f1 - f0) * t * t / (2 * dur))
    env = np.minimum(t / 0.002, 1.0) * np.exp(-t / 0.018)
    mono = np.sin(phase) * env
    mono = mono / (np.abs(mono).max() or 1.0) * 0.55
    return (np.stack([mono, mono], axis=1) * 32767).astype(np.int16)


class Sfx:
    """One-shot UI sounds; every call is a safe no-op without a mixer."""

    def __init__(self, enabled: bool = True, volume: float = 0.5) -> None:
        self.enabled = enabled
        self.volume = volume
        self._click = None
        self._built = False

    def _build(self) -> None:
        if self._built or not self.enabled:
            return
        self._built = True
        try:
            import pygame

            self._click = pygame.sndarray.make_sound(click_samples())
            self._click.set_volume(self.volume)
        except Exception:
            self.enabled = False

    def click(self) -> None:
        if not self.enabled:
            return
        self._build()
        if self._click is not None:
            try:
                self._click.play()
            except Exception:
                self.enabled = False

    def set_volume(self, v: float) -> None:
        self.volume = max(0.0, min(1.0, v))
        if self._click is not None:
            try:
                self._click.set_volume(self.volume)
            except Exception:
                pass
