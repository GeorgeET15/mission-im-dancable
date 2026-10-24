"""Elapsed-time playback position for reference-skeleton animation.

Driven by wall-clock time so playback speed matches the original video on
any machine, independent of render frame rate.
"""

from __future__ import annotations


class PlaybackClock:
    """Tracks position (seconds) inside a clip of known duration."""

    def __init__(self, duration: float, playing: bool = True) -> None:
        if duration <= 0:
            raise ValueError("duration must be positive")
        self.duration = float(duration)
        self.playing = playing
        self._t = 0.0
        self._last_wall: float | None = None

    @property
    def time(self) -> float:
        return self._t

    @property
    def finished(self) -> bool:
        return self._t >= self.duration and not self.playing

    def play(self) -> None:
        self.playing = True
        self._last_wall = None

    def pause(self) -> None:
        self.playing = False

    def toggle(self) -> None:
        if self.playing:
            self.pause()
        else:
            self.play()

    def restart(self) -> None:
        self._t = 0.0
        self.play()

    def seek(self, t: float) -> None:
        """Jump to an absolute time, clamped to [0, duration]."""
        self._t = min(max(float(t), 0.0), self.duration)

    def skip(self, dt: float) -> None:
        """Jump forward/backward by dt seconds."""
        self.seek(self._t + dt)

    def now(self, wall: float) -> float:
        """Advance by elapsed wall time and return the position.

        Playback auto-pauses at the end of the clip; call restart() or
        seek() to continue.
        """
        if self.playing and self._last_wall is not None:
            self._t = min(self._t + (wall - self._last_wall), self.duration)
            if self._t >= self.duration:
                self.playing = False
        self._last_wall = wall
        return self._t
