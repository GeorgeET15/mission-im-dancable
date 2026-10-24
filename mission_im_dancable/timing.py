"""Frame-rate measurement for the live loop."""

from __future__ import annotations

import time
from collections import deque


class FpsCounter:
    """Smoothed FPS over the most recent frames."""

    def __init__(self, window: int = 30) -> None:
        if window <= 0:
            raise ValueError("window must be positive")
        self._times: deque[float] = deque(maxlen=window)

    def tick(self, now: float | None = None) -> float:
        """Record a frame and return the smoothed FPS (0.0 on first call)."""
        now = time.perf_counter() if now is None else now
        self._times.append(now)
        if len(self._times) < 2:
            return 0.0
        span = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / span if span > 0 else 0.0
