"""Move-by-move practice mode: reference vs. live skeleton, side-by-side."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import torch

from .camera import CameraError, open_camera
from .config import AppConfig, resolve_device
from .moves import Move, MovesFileError, default_moves, load_moves
from .pose import PoseTracker
from .reference import (
    ReferenceChoreography,
    ReferenceVideoError,
    load_or_extract,
)
from .render import draw_pose
from .scoring import (
    CorrectionTracker,
    ScoreSmoother,
    best_window_score,
    describe_correction,
    mirror_keypoints,
    worst_mismatch,
)
from .timing import FpsCounter

WINDOW_NAME = "Mission Im-Dancable — practice"
PANEL_BG = (24, 20, 16)  # dark mission-control backdrop
TEXT = (220, 220, 220)
ACCENT = (0, 220, 180)


class PracticeSession:
    """Tracks which move is active and the playback position inside it.

    The reference pose loops inside the active move's [start, end] range.
    """

    def __init__(
        self,
        choreo: ReferenceChoreography,
        moves: list[Move],
        window_s: float = 0.15,
    ) -> None:
        if not moves:
            raise ValueError("PracticeSession needs at least one move")
        self.choreo = choreo
        self.moves = moves
        self.window_s = window_s
        self.move_index = 0
        self.time = moves[0].start

    @property
    def move(self) -> Move:
        return self.moves[self.move_index]

    def set_move(self, index: int) -> None:
        self.move_index = max(0, min(index, len(self.moves) - 1))
        self.time = self.move.start

    def advance(self, dt: float) -> None:
        """Advance playback by dt seconds, looping inside the active move."""
        self.time += dt
        move = self.move
        if self.time > move.end:
            span = max(move.duration, 1e-6)
            self.time = move.start + (self.time - move.start) % span

    def _window_indices(self) -> np.ndarray:
        """Sample indices whose timestamps fall inside the match window."""
        ts = self.choreo.timestamps
        lo = np.searchsorted(ts, self.time - self.window_s)
        hi = np.searchsorted(ts, self.time + self.window_s, side="right")
        return np.arange(lo, hi)

    def window_keypoints(self) -> np.ndarray:
        return self.choreo.keypoints[self._window_indices()]

    def nearest_keypoints(self) -> np.ndarray | None:
        """Closest sampled reference pose to the current time (for display)."""
        ts = self.choreo.timestamps
        if ts.size == 0:
            return None
        i = int(np.searchsorted(ts, self.time))
        candidates = [j for j in (i - 1, i) if 0 <= j < ts.size]
        if not candidates:
            return None
        best = min(candidates, key=lambda j: abs(ts[j] - self.time))
        return self.choreo.keypoints[best]


def _scaled_panel_pose(
    keypoints: np.ndarray, src_w: int, src_h: int, dst_w: int, dst_h: int
) -> np.ndarray:
    """Scale keypoints from video coordinates into the reference panel."""
    scale = min(dst_w / src_w, dst_h / src_h) * 0.9 if src_w and src_h else 1.0
    off_x = (dst_w - src_w * scale) / 2
    off_y = (dst_h - src_h * scale) / 2
    out = keypoints.copy()
    out[:, 0] = out[:, 0] * scale + off_x
    out[:, 1] = out[:, 1] * scale + off_y
    return out


def _draw_text(img: np.ndarray, lines: list[tuple[str, tuple[int, int, int], float]], x: int = 10, y: int = 26, step: int = 22) -> None:
    for i, (text, color, thick) in enumerate(lines):
        cv2.putText(img, text, (x, y + i * step), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, int(thick), cv2.LINE_AA)


def run_practice(cfg: AppConfig) -> int:
    """Reference-choreography practice loop. Returns an exit code."""
    device = resolve_device(cfg.device, torch.cuda.is_available())
    tracker = PoseTracker(cfg.model_path, device, cfg.confidence)

    try:
        choreo = load_or_extract(
            tracker,
            cfg.source,
            cfg.sample_fps,
            cfg.cache_dir,
            cfg.confidence,
            model_name=cfg.model_path.name,
        )
    except (ReferenceVideoError, FileNotFoundError, OSError) as exc:
        print(f"Error: {exc}")
        return 2

    try:
        moves = (
            load_moves(cfg.moves_path, choreo.duration)
            if cfg.moves_path
            else default_moves(choreo.duration)
        )
    except MovesFileError as exc:
        print(f"Error: {exc}")
        return 2

    try:
        cap = open_camera(cfg)
    except CameraError as exc:
        print(f"Error: {exc}")
        return 2

    session = PracticeSession(choreo, moves, cfg.window_s)
    smoother = ScoreSmoother()
    corrections = CorrectionTracker()
    fps = FpsCounter()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    print(
        f"Practice on {device.upper()} — {len(moves)} move(s). "
        "N next move, P previous, Q/Esc quit.",
        flush=True,
    )
    last = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Error: failed to read a frame from the camera.")
                return 3

            now = time.perf_counter()
            session.advance(now - last)
            last = now

            live = tracker.predict(frame)
            ref_window = session.window_keypoints()
            if cfg.mirror:
                ref_window = np.stack(
                    [mirror_keypoints(r, choreo.width) for r in ref_window]
                )

            score_text = "match --"
            correction_text = ""
            if live is not None and ref_window.size:
                best = best_window_score(live.keypoints, ref_window, cfg.confidence)
                if best is not None:
                    pose_score, best_i = best
                    smoothed = smoother.update(pose_score.score)
                    score_text = f"match {smoothed * 100:3.0f}%"
                    idx = corrections.update(worst_mismatch(pose_score))
                    if idx is not None:
                        correction_text = describe_correction(
                            idx, live.keypoints, ref_window[best_i]
                        )
                else:
                    smoother.update(None)
                    corrections.update(None)
                    score_text = "match -- (low confidence)"
            else:
                smoother.update(None)
                corrections.update(None)

            h, w = frame.shape[:2]
            left = np.full((h, w, 3), PANEL_BG, dtype=np.uint8)
            ref_pose = session.nearest_keypoints()
            move = session.move
            if ref_pose is not None and ref_pose[:, 2].max() > 0:
                shown = (
                    mirror_keypoints(ref_pose, choreo.width) if cfg.mirror else ref_pose
                )
                draw_pose(
                    left,
                    _scaled_panel_pose(shown, choreo.width, choreo.height, w, h),
                    cfg.confidence,
                )
            progress = (
                (session.time - move.start) / move.duration if move.duration else 0.0
            )
            _draw_text(
                left,
                [
                    (f"MISSION {session.move_index + 1}/{len(moves)}: {move.name}", TEXT, 2),
                    (f"{move.start:.1f}s - {move.end:.1f}s  |  loop {progress * 100:3.0f}%", ACCENT, 1),
                    ("REFERENCE", TEXT, 1),
                ],
            )

            right = frame
            if live is not None:
                draw_pose(right, live.keypoints, cfg.confidence)
            _draw_text(
                right,
                [
                    ("YOU", TEXT, 2),
                    (score_text, ACCENT, 2),
                    (correction_text or "match the pose", (60, 60, 255), 1),
                    (f"FPS {fps.tick():4.1f} | N next  P prev  Q quit", TEXT, 1),
                ],
            )

            cv2.imshow(WINDOW_NAME, np.hstack([left, right]))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("n"), ord("N")):
                session.set_move(session.move_index + 1)
            elif key in (ord("p"), ord("P")):
                session.set_move(session.move_index - 1)
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0
