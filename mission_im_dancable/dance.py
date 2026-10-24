"""Dance definition shared by Learn Mode and Play Mode.

A Dance ties a reference video to its metadata and ordered move segments.
It is built after a video is analyzed (poses extracted + duration known);
the annotation file itself lives in moves.py's ``<stem>.moves.json`` format.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .moves import Move
from .reference import ReferenceChoreography


@dataclass(frozen=True)
class Dance:
    """One entry in the dance library."""

    id: str
    title: str
    video_path: Path
    duration: float
    video_fps: float
    moves: list[Move]
    difficulty: str = ""


def build_dance(
    video_path: Path,
    choreo: ReferenceChoreography,
    moves: list[Move],
    meta: dict | None = None,
) -> Dance:
    """Assemble a Dance from an analyzed video + parsed annotation metadata."""
    meta = meta or {}
    return Dance(
        id=meta.get("id") or video_path.stem,
        title=meta.get("title") or video_path.stem,
        video_path=video_path,
        duration=choreo.duration,
        video_fps=choreo.video_fps,
        moves=moves,
        difficulty=meta.get("difficulty", ""),
    )
