"""Learn Mode session: free-form study of a dance without time pressure.

Pygame-free so it can be unit-tested headless. Owns the playhead position,
playback speed, selected segment, and loop mode; the screen layer feeds it
dt and renders whatever state it reports. No scoring, no combo, no fail.
"""

from __future__ import annotations

from .moves import Move, move_index_at

SPEEDS = (0.25, 0.5, 0.75, 1.0)
LOOP_MODES = ("off", "segment", "all")
SEEK_STEP = 2.0  # seconds per arrow-key seek


class LearnSession:
    """Playback state for studying one dance.

    ``time`` is always inside [0, duration]; seeking and looping clamp or
    wrap exactly, so position never drifts.
    """

    def __init__(self, moves: list[Move], duration: float, video_fps: float = 30.0) -> None:
        if not moves:
            raise ValueError("LearnSession needs at least one move")
        if duration <= 0:
            raise ValueError("duration must be positive")
        self.moves = moves
        self.duration = float(duration)
        self.video_fps = video_fps if video_fps > 0 else 30.0
        self.time = moves[0].start
        self.playing = False
        self.speed_index = SPEEDS.index(1.0)
        self.selected = 0
        self.loop = "off"

    # -- properties ----------------------------------------------------------

    @property
    def speed(self) -> float:
        return SPEEDS[self.speed_index]

    @property
    def frame_dt(self) -> float:
        return 1.0 / self.video_fps

    @property
    def active_index(self) -> int:
        """Segment under the playhead (gaps belong to the upcoming move)."""
        return move_index_at(self.moves, self.time)

    @property
    def finished(self) -> bool:
        return not self.playing and self.time >= self.duration

    # -- transport -------------------------------------------------------------

    def advance(self, dt: float) -> None:
        """Advance the playhead by dt render seconds × playback speed."""
        if self.playing:
            self._set(self.time + dt * self.speed)

    def _set(self, t: float) -> None:
        seg = self.moves[self.selected]
        if self.loop == "segment" and t >= seg.end:
            # wrap inside the selected segment — exact arithmetic, no drift
            self.time = seg.start + (t - seg.end) % max(seg.duration, 1e-6)
        elif self.loop == "all" and t >= self.duration:
            self.time = t % self.duration
        else:
            self.time = min(t, self.duration)
            if self.time >= self.duration:
                self.playing = False

    def seek(self, t: float) -> None:
        """Jump the playhead to t seconds, clamped to the dance."""
        self.time = min(max(float(t), 0.0), self.duration)

    def skip(self, dt: float) -> None:
        self.seek(self.time + dt)

    def toggle(self) -> None:
        if self.playing:
            self.playing = False
        elif self.time >= self.duration:
            self.time = 0.0  # replay from the top after the dance ended
            self.playing = True
        else:
            self.playing = True

    def step_frames(self, n: int) -> None:
        """Step n reference-video frames; pauses so single-stepping holds."""
        self.playing = False
        self.seek(self.time + n * self.frame_dt)

    # -- speed ------------------------------------------------------------------

    def cycle_speed(self, direction: int = 1) -> float:
        self.speed_index = (self.speed_index + direction) % len(SPEEDS)
        return self.speed

    def set_speed(self, speed: float) -> None:
        """Snap to the nearest supported speed."""
        self.speed_index = min(
            range(len(SPEEDS)), key=lambda i: abs(SPEEDS[i] - speed)
        )

    # -- segments -----------------------------------------------------------------

    def select(self, index: int) -> int:
        """Select a segment and jump the playhead to its start."""
        self.selected = max(0, min(index, len(self.moves) - 1))
        self.seek(self.moves[self.selected].start)
        return self.selected

    def next_segment(self) -> int:
        return self.select(self.selected + 1)

    def prev_segment(self) -> int:
        return self.select(self.selected - 1)

    def cycle_loop(self) -> str:
        """Loop: off → selected segment → whole dance → off."""
        self.loop = LOOP_MODES[(LOOP_MODES.index(self.loop) + 1) % len(LOOP_MODES)]
        return self.loop

    def segment_state(self, index: int) -> str:
        """Timeline coloring: done / selected / active / upcoming."""
        if index == self.selected:
            return "selected"
        if self.moves[index].end <= self.time:
            return "done"
        if index == self.active_index:
            return "active"
        return "upcoming"
