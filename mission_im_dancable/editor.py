"""Segment editing: a working draft of a dance's move list plus optional
auto-suggested boundaries from the beat grid and pose-motion data.

Pygame-free so it can be unit-tested. The draft only touches Move objects;
saving goes through moves.save_dance_file, which validates against the real
video duration and preserves existing metadata.
"""

from __future__ import annotations

import numpy as np

from .beats import BeatGrid
from .moves import Move, validate_segments
from .reference import ReferenceChoreography

MIN_SEGMENT = 0.2  # seconds — no point slicing thinner than this
BEATS_PER_PHRASE = 8
MIN_GAP = 0.5  # boundaries this close to an edge are dropped


class SegmentDraft:
    """Editable working copy of a dance's segments.

    ``from_draft`` marks segments that came from automatic suggestions
    rather than the user's own file — the UI renders them differently and
    saving confirms them into real annotations.
    """

    def __init__(self, moves: list[Move], duration: float) -> None:
        self.duration = duration
        self.segments: list[Move] = list(moves)
        self.selected = 0
        self.from_draft = False
        self.dirty = False

    # -- helpers -------------------------------------------------------------

    @property
    def segment(self) -> Move:
        return self.segments[self.selected]

    def _set(self, index: int, move: Move) -> None:
        self.segments[index] = move
        self.dirty = True

    def select(self, index: int) -> int:
        self.selected = max(0, min(index, len(self.segments) - 1))
        return self.selected

    def accept(self) -> None:
        """Confirm suggested boundaries as the user's own annotations."""
        self.from_draft = False

    def set_segments(self, moves: list[Move], from_draft: bool = False) -> None:
        self.segments = list(moves)
        self.selected = min(self.selected, max(len(moves) - 1, 0))
        self.from_draft = from_draft
        self.dirty = True

    # -- edits -----------------------------------------------------------------

    def split_at(self, t: float) -> bool:
        """Split the segment containing t. False when nothing to split or
        the cut would leave a sub-MIN_SEGMENT piece."""
        for i, m in enumerate(self.segments):
            if m.start + MIN_SEGMENT <= t <= m.end - MIN_SEGMENT:
                a = Move(m.name, m.start, t, m.instruction)
                b = Move(m.name + " (2)" if m.name else "", t, m.end, "")
                self.segments[i : i + 1] = [a, b]
                self.selected = i
                self.dirty = True
                return True
        return False

    def merge_next(self) -> bool:
        """Remove the boundary after the selected segment (merge it with
        the next one). False when nothing follows."""
        i = self.selected
        if i + 1 >= len(self.segments):
            return False
        a, b = self.segments[i], self.segments[i + 1]
        self.segments[i : i + 2] = [
            Move(a.name, a.start, b.end, a.instruction or b.instruction)
        ]
        self.dirty = True
        return True

    def nudge_boundary(self, dt: float) -> bool:
        """Move the boundary after the selected segment by dt seconds.

        When the two segments are contiguous both edges move together;
        otherwise only the selected segment's end moves (the gap shifts).
        """
        i = self.selected
        if i + 1 >= len(self.segments):
            return False
        a, b = self.segments[i], self.segments[i + 1]
        new_end = a.end + dt
        contiguous = abs(b.start - a.end) < 1e-6
        if new_end <= a.start + MIN_SEGMENT:
            return False
        if contiguous and new_end >= b.end - MIN_SEGMENT:
            return False
        self._set(i, Move(a.name, a.start, new_end, a.instruction))
        if contiguous:
            self.segments[i + 1] = Move(b.name, new_end, b.end, b.instruction)
        return True

    def reorder(self, direction: int) -> bool:
        """Swap the selected segment with its neighbour in the list order.

        Time ranges travel with the segment; validation at save time
        rejects any overlap this creates — reorder is a rename/order tool,
        not a time machine.
        """
        j = self.selected + direction
        if not (0 <= j < len(self.segments)):
            return False
        self.segments[self.selected], self.segments[j] = (
            self.segments[j],
            self.segments[self.selected],
        )
        self.selected = j
        self.dirty = True
        return True

    def rename(self, name: str) -> None:
        m = self.segment
        self._set(self.selected, Move(name.strip() or m.name, m.start, m.end, m.instruction))

    # -- validation ------------------------------------------------------------

    def errors(self) -> list[str]:
        return validate_segments(self.segments, self.duration)


# -- automatic suggestions -----------------------------------------------------


def _beat_boundaries(beats: np.ndarray, duration: float) -> list[float]:
    """Every BEATS_PER_PHRASE-th beat, trimmed away from the edges."""
    return [
        float(b)
        for b in beats[BEATS_PER_PHRASE - 1 :: BEATS_PER_PHRASE]
        if MIN_GAP < float(b) < duration - MIN_GAP
    ]


def _motion_boundaries(choreo: ReferenceChoreography, duration: float) -> list[float]:
    """Valleys in pose-motion energy — sustained stillness between moves."""
    kp = choreo.keypoints
    if kp.shape[0] < 4:
        return []
    conf = kp[:, :, 2] > 0
    diffs = np.linalg.norm(kp[1:, :, :2] - kp[:-1, :, :2], axis=2)
    valid = conf[1:] & conf[:-1]
    cnt = valid.sum(axis=1)
    energy = np.where(
        cnt >= 4,
        (diffs * valid).sum(axis=1) / np.maximum(cnt, 1),
        np.nan,
    )
    ts = choreo.timestamps[1 : 1 + energy.size]
    finite = np.isfinite(energy)
    if finite.sum() < 4 or np.nanmax(energy[finite]) <= 0:
        return []
    thresh = np.nanpercentile(energy[finite], 30)
    bounds: list[float] = []
    last = -6.0
    for i in range(energy.size):
        if (
            finite[i]
            and energy[i] <= thresh
            and ts[i] - last >= 1.5
            and MIN_GAP < ts[i] < duration - MIN_GAP
        ):
            bounds.append(float(ts[i]))
            last = ts[i]
    return bounds


def suggest_boundaries(
    choreo: ReferenceChoreography,
    beats: BeatGrid | None,
    duration: float,
) -> list[float]:
    """Candidate segment boundaries from the video's own data.

    Beats win when the audio produced a usable grid (dance moves live on
    musical phrases); pose-motion valleys are the fallback for audio-free
    or irregular clips. Never returns edges.
    """
    if beats is not None and beats.beats.size >= BEATS_PER_PHRASE:
        bounds = _beat_boundaries(beats.beats, duration)
        if bounds:
            return bounds
    return _motion_boundaries(choreo, duration)


def draft_moves(duration: float, boundaries: list[float]) -> list[Move]:
    """Named suggested segments from boundary timestamps.

    Boundaries too close to a neighbour are dropped so a draft is always
    a valid, non-overlapping, non-zero-length segment list.
    """
    edges = [0.0]
    for b in sorted(boundaries):
        if b - edges[-1] >= MIN_SEGMENT:
            edges.append(float(b))
    edges.append(duration)
    if edges[-2] != 0.0 and edges[-1] - edges[-2] < MIN_SEGMENT:
        edges.pop(-2)  # thin tail merges into the last segment
    return [
        Move(f"phrase {i + 1}", edges[i], edges[i + 1])
        for i in range(len(edges) - 1)
    ]
