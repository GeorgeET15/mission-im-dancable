"""Manual move segmentation loaded from a JSON config file.

Format (dance annotation file ``<video-stem>.moves.json``):

    {
      "id": "my-dance",                // optional — defaults to the video stem
      "title": "My dance",             // optional — display title
      "difficulty": "medium",          // optional — free text
      "moves": [
        {"name": "intro sway", "start": 0.0, "end": 2.4,
         "instruction": "shift weight left to right"},   // instruction optional
        {"name": "step touch", "start": 2.4, "end": 6.0}
      ]
    }

Timestamps are seconds within the reference video. Segment boundaries are
written by hand against the real video — nothing here invents them.
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
    instruction: str = ""  # optional practice note for Learn Mode

    @property
    def duration(self) -> float:
        return self.end - self.start


def default_moves(duration: float) -> list[Move]:
    """When no moves file is given, the whole video is one move."""
    return [Move(name="full choreography", start=0.0, end=duration)]


def load_moves(path: Path, video_duration: float) -> list[Move]:
    """Parse and validate a moves JSON file against the video duration."""
    moves, _ = load_dance_file(path, video_duration)
    return moves


def load_dance_file(path: Path, video_duration: float) -> tuple[list[Move], dict]:
    """Parse a dance annotation file into (moves, metadata).

    Metadata holds the optional free-text fields ``id``, ``title`` and
    ``difficulty``; non-string values are ignored so metadata can never
    break an otherwise valid file.
    """
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

    meta = {
        key: raw[key].strip()
        for key in ("id", "title", "difficulty")
        if isinstance(raw.get(key), str) and raw[key].strip()
    }
    moves: list[Move] = []
    for i, item in enumerate(items):
        moves.append(_parse_move(item, i, video_duration, path))
    return moves, meta


def load_dance_meta(path: Path) -> dict:
    """Lenient metadata read for an existing annotation file.

    Returns the optional id/title/difficulty fields, or {} when the file is
    missing, unreadable, or not valid JSON — meta is never fatal.
    """
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {
        key: raw[key].strip()
        for key in ("id", "title", "difficulty")
        if isinstance(raw.get(key), str) and raw[key].strip()
    }


def validate_segments(moves: list[Move], duration: float) -> list[str]:
    """Errors for a proposed segment list; empty list means saveable.

    Segments may have gaps between them (time belongs to the upcoming
    segment) but must never overlap or run outside the video.
    """
    errors: list[str] = []
    if not moves:
        return ["at least one segment is required"]
    prev_end = -1.0
    for i, m in enumerate(moves):
        label = f"segment {i + 1} ('{m.name}')"
        if not m.name.strip():
            errors.append(f"{label}: name must not be empty")
        if m.start < 0:
            errors.append(f"{label}: starts before the video")
        if m.end <= m.start:
            errors.append(f"{label}: zero or negative length")
        if m.end > duration + TIMESTAMP_EPSILON:
            errors.append(f"{label}: ends past the video duration")
        if m.start < prev_end - TIMESTAMP_EPSILON:
            errors.append(f"{label}: overlaps the previous segment")
        prev_end = max(prev_end, m.end)
    return errors


def save_dance_file(path: Path, moves: list[Move], meta: dict | None = None) -> None:
    """Write a ``<stem>.moves.json`` sidecar.

    Metadata (id/title/difficulty) already in the file is preserved unless
    overridden by ``meta`` — saving edits never destroys unrelated fields.
    """
    merged = {**load_dance_meta(path), **(meta or {})}
    raw = dict(merged)
    raw["moves"] = [
        {"name": m.name, "start": round(m.start, 3), "end": round(m.end, 3),
         **({"instruction": m.instruction} if m.instruction else {})}
        for m in moves
    ]
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(raw, indent=2) + "\n")
    tmp.replace(path)


def move_index_at(moves: list[Move], t: float) -> int:
    """Segment index owning time t: the first move whose end is past t.

    Times inside a gap between segments belong to the *upcoming* move —
    the one the dancer should already be moving into.
    """
    for i, move in enumerate(moves):
        if t < move.end:
            return i
    return len(moves) - 1


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
    instruction = item.get("instruction", "")
    if not isinstance(instruction, str):
        instruction = ""
    return Move(
        name=name.strip(),
        start=start,
        end=min(end, video_duration),
        instruction=instruction.strip(),
    )
