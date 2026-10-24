"""Manual move segmentation loaded from a JSON config file.

Format:

    {
      "moves": [
        {"name": "intro sway", "start": 0.0, "end": 2.4},
        {"name": "step touch", "start": 2.4, "end": 6.0}
      ]
    }

Timestamps are seconds within the reference video.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

TIMESTAMP_EPSILON = 1e-3  # float slack for end <= duration checks


class MovesFileError(RuntimeError):
    pass


@dataclass(frozen=True)
class Move:
    name: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def default_moves(duration: float) -> list[Move]:
    """When no moves file is given, the whole video is one move."""
    return [Move(name="full choreography", start=0.0, end=duration)]


def load_moves(path: Path, video_duration: float) -> list[Move]:
    """Parse and validate a moves JSON file against the video duration."""
    try:
        raw = json.loads(path.read_text())
    except OSError as exc:
        raise MovesFileError(f"Cannot read moves file {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise MovesFileError(f"Moves file {path} is not valid JSON: {exc}") from exc

    items = raw.get("moves") if isinstance(raw, dict) else None
    if not isinstance(items, list) or not items:
        raise MovesFileError(
            f"Moves file {path} must contain a non-empty 'moves' list, "
            'e.g. {"moves": [{"name": "step", "start": 0, "end": 2.5}]}'
        )

    moves: list[Move] = []
    for i, item in enumerate(items):
        moves.append(_parse_move(item, i, video_duration, path))
    return moves


def _parse_move(item: object, index: int, video_duration: float, path: Path) -> Move:
    label = f"move #{index + 1} in {path}"
    if not isinstance(item, dict):
        raise MovesFileError(f"{label} must be an object with name/start/end")
    name = item.get("name")
    start = item.get("start")
    end = item.get("end")
    if not isinstance(name, str) or not name.strip():
        raise MovesFileError(f"{label} needs a non-empty string 'name'")
    if not isinstance(start, (int, float)) or isinstance(start, bool):
        raise MovesFileError(f"{label} ('{name}') needs a numeric 'start'")
    if not isinstance(end, (int, float)) or isinstance(end, bool):
        raise MovesFileError(f"{label} ('{name}') needs a numeric 'end'")
    start, end = float(start), float(end)
    if start < 0:
        raise MovesFileError(f"{label} ('{name}'): start {start:g}s is before the video begins")
    if end <= start:
        raise MovesFileError(f"{label} ('{name}'): end {end:g}s must be after start {start:g}s")
    if end > video_duration + TIMESTAMP_EPSILON:
        raise MovesFileError(
            f"{label} ('{name}'): end {end:g}s exceeds the video duration "
            f"({video_duration:g}s)"
        )
    return Move(name=name.strip(), start=start, end=min(end, video_duration))
