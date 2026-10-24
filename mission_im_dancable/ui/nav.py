"""Pygame-free screen navigation: stack, transitions, input delegation.

Screens are duck-typed: enter(ctx), exit(ctx), handle_event(ctx, ev),
update(ctx, dt), draw(ctx, surf). Tested with fake screens — no pygame
or display required.
"""

from __future__ import annotations

FADE_SECONDS = 0.25


class Navigator:
    """Stack-based screen manager with a short fade + input lock."""

    def __init__(self, fade: float = FADE_SECONDS) -> None:
        self._stack: list = []
        self._fade = 0.0
        self._fade_duration = fade

    @property
    def current(self):
        return self._stack[-1] if self._stack else None

    @property
    def depth(self) -> int:
        return len(self._stack)

    @property
    def transition_active(self) -> bool:
        return self._fade > 0.0

    def push(self, screen) -> None:
        if self.current is not None:
            self.current.exit()
        self._stack.append(screen)
        screen.enter()
        self._fade = self._fade_duration

    def pop(self):
        if not self._stack:
            return None
        old = self._stack.pop()
        old.exit()
        if self.current is not None:
            self.current.enter()
        self._fade = self._fade_duration
        return old

    def swap(self, screen) -> None:
        """Pop the top screen and push a new one without re-entering the
        revealed screen. Used for A→B handoffs inside the same flow
        (ready check → gameplay) where the screen underneath stays
        suspended rather than becoming active in between."""
        if self._stack:
            self._stack.pop().exit()
        self._stack.append(screen)
        screen.enter()
        self._fade = self._fade_duration

    def replace(self, screen) -> None:
        while self._stack:
            self._stack.pop().exit()
        self._stack.append(screen)
        screen.enter()
        self._fade = self._fade_duration

    def handle_event(self, ctx, ev) -> None:
        # Lock input during transitions so one keypress can't fire two
        # state changes on consecutive screens.
        if self.transition_active or self.current is None:
            return
        self.current.handle_event(ctx, ev)

    def update(self, ctx, dt: float) -> None:
        self._fade = max(0.0, self._fade - dt)
        if self.current is not None:
            self.current.update(ctx, dt)

    def draw(self, ctx, surf) -> None:
        if self.current is not None:
            self.current.draw(ctx, surf)
