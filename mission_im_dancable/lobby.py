"""Looping synth lobby music for the menu screens.

The track is generated in-code (numpy → int16 stereo) so the game ships no
audio asset and carries no licensing concerns. It plays through a dedicated
``pygame.mixer.Channel`` — *not* ``pygame.mixer.music``, which the dance
soundtrack owns — so lobby music and dance music never fight over the same
stream. A mellow 96 BPM arpeggio over Am–F–C–G, about 10 s per loop.
"""

from __future__ import annotations

import numpy as np

_SR = 44100
_BPM = 96.0
_EIGHTH = 60.0 / _BPM / 2.0  # 0.3125 s


def _freq(midi: int) -> float:
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


def _pluck(t: np.ndarray, midi: int, decay: float = 0.28) -> np.ndarray:
    """Sine + 3rd-harmonic pluck with a fast attack and exp decay."""
    f = _freq(midi)
    body = np.sin(2 * np.pi * f * t) + 0.35 * np.sin(2 * np.pi * f * 3.01 * t)
    env = np.minimum(t / 0.006, 1.0) * np.exp(-t / decay)
    return body * env


def lobby_samples(sr: int = _SR) -> np.ndarray:
    """One loop of the lobby track as an int16 stereo (N, 2) array."""
    # root midi, arpeggio notes (chord tones + octave colour)
    bars = [
        (45, [57, 60, 64, 69]),  # A minor
        (41, [53, 57, 60, 65]),  # F major
        (48, [52, 55, 60, 64]),  # C major
        (43, [47, 50, 55, 59]),  # G major
    ]
    bar_len = 8 * _EIGHTH
    total = int(len(bars) * bar_len * sr)
    out = np.zeros(total, dtype=np.float64)
    for bi, (root, notes) in enumerate(bars):
        t0 = bi * bar_len
        # bass: root note held for the bar
        i0 = int(t0 * sr)
        tt = np.arange(total - i0) / sr
        out[i0:] += 0.22 * _pluck(tt, root - 12, decay=1.4)
        # arpeggio: eighth notes cycling the chord tones
        for ei in range(8):
            start = int((t0 + ei * _EIGHTH) * sr)
            if start >= total:
                break
            tt = np.arange(total - start) / sr
            out[start:] += 0.16 * _pluck(tt, notes[ei % len(notes)])
    peak = np.max(np.abs(out)) or 1.0
    mono = np.clip(out / peak * 0.75, -1.0, 1.0)
    stereo = np.stack([mono, np.roll(mono, int(0.012 * sr))], axis=1)  # slight width
    # soft fade at both ends so the loop point doesn't click — applied
    # after the stereo roll so both channels start/end silent
    fade = int(0.05 * sr)
    ramp = np.linspace(0.0, 1.0, fade)[:, None]
    stereo[:fade] *= ramp
    stereo[-fade:] *= ramp[::-1]
    return (stereo * 32767).astype(np.int16)


class LobbyMusic:
    """Menu music loop on a reserved mixer channel.

    `ensure()` starts the loop if it isn't already playing; `stop()` ends
    it. Safe no-ops when the mixer is unavailable or disabled — screens just
    call it every frame.
    """

    def __init__(self, enabled: bool = True, volume: float = 0.35) -> None:
        self.enabled = enabled
        self.volume = volume
        self._sound = None
        self._channel = None
        self._built = False

    def _build(self) -> None:
        if self._built or not self.enabled:
            return
        self._built = True
        try:
            import pygame

            self._sound = pygame.sndarray.make_sound(lobby_samples())
        except Exception:
            self.enabled = False  # no mixer / no numpy path — stay silent

    def ensure(self) -> None:
        if not self.enabled:
            return
        self._build()
        if self._sound is None:
            return
        if self._channel is None or not self._channel.get_busy():
            try:
                self._channel = self._sound.play(loops=-1)
                if self._channel is not None:
                    self._channel.set_volume(self.volume)
            except Exception:
                self.enabled = False

    def stop(self) -> None:
        if self._channel is not None:
            try:
                self._channel.stop()
            except Exception:
                pass
            self._channel = None

    def set_volume(self, v: float) -> None:
        self.volume = max(0.0, min(1.0, v))
        if self._channel is not None:
            try:
                self._channel.set_volume(self.volume)
            except Exception:
                pass
