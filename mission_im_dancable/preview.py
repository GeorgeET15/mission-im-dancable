"""Reference-skeleton preview: animated playback of extracted poses.

Plays the ReferenceChoreography at the original video speed using elapsed
wall time, next to the live camera skeleton. ENTER hands the same camera
straight into move-by-move practice.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from .camera import CameraError, open_camera
from .config import AppConfig
from .moves import MovesFileError
from .playback import PlaybackClock
from .practice import _draw_text, _scaled_panel_pose, prepare, run_practice_loop
from .reference import ReferenceChoreography, ReferenceVideoError, pose_at
from .render import draw_pose
from .scoring import mirror_keypoints
from .timing import FpsCounter

WINDOW_NAME = "Mission Im-Dancable — reference preview"
PANEL_BG = (24, 20, 16)
TEXT = (220, 220, 220)
ACCENT = (0, 220, 180)
WARN = (60, 60, 255)
SEEK_STEP = 2.0  # seconds per arrow press


def reference_panel(
    choreo: ReferenceChoreography,
    keypoints: np.ndarray | None,
    t: float,
    clock_playing: bool,
    width: int,
    height: int,
    conf: float,
    mirrored: bool,
) -> np.ndarray:
    """Render the animated reference skeleton panel for time t."""
    panel = np.full((height, width, 3), PANEL_BG, dtype=np.uint8)
    shown = None
    if keypoints is not None and keypoints[:, 2].max() > 0:
        shown = mirror_keypoints(keypoints, choreo.width) if mirrored else keypoints
        draw_pose(
            panel,
            _scaled_panel_pose(shown, choreo.width, choreo.height, width, height),
            conf,
        )
    frac = t / choreo.duration if choreo.duration else 0.0
    state = "ENDED" if (not clock_playing and t >= choreo.duration) else (
        "PLAYING" if clock_playing else "PAUSED"
    )
    _draw_text(
        panel,
        [
            ("REFERENCE PREVIEW" + (" (MIRRORED)" if mirrored else ""), TEXT, 2),
            (f"t={t:4.1f}s / {choreo.duration:4.1f}s  |  {state}", ACCENT, 1),
        ],
    )
    if shown is None:
        _draw_text(panel, [("no pose detected at this time", WARN, 1)], y=96)
    # progress bar along the bottom
    bar_h = 10
    cv2.rectangle(panel, (8, height - 18), (width - 8, height - 18 + bar_h), (60, 60, 60), -1)
    cv2.rectangle(
        panel,
        (8, height - 18),
        (8 + int((width - 16) * min(max(frac, 0.0), 1.0)), height - 18 + bar_h),
        ACCENT,
        -1,
    )
    return panel


def run_preview(cfg: AppConfig) -> int:
    """Playback loop. Returns an exit code; ENTER continues into practice."""
    try:
        tracker, choreo, moves, device = prepare(cfg)
    except (ReferenceVideoError, MovesFileError, FileNotFoundError, OSError) as exc:
        print(f"Error: {exc}")
        return 2

    cap = None
    if not cfg.no_cam:
        try:
            cap = open_camera(cfg)
        except CameraError as exc:
            print(f"Warning: {exc} — running preview without a camera.", flush=True)

    clock = PlaybackClock(choreo.duration)
    fps = FpsCounter()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    print(
        f"Preview on {device.upper()} — SPACE pause · ←/→ seek {SEEK_STEP:g}s · "
        "R restart · ENTER start practice · Q quit.",
        flush=True,
    )
    start_practice = False
    try:
        while True:
            t = clock.now(time.perf_counter())
            pose = pose_at(choreo, t)

            cam_frame, live = None, None
            if cap is not None:
                ok, cam_frame = cap.read()
                if ok:
                    live = tracker.predict(cam_frame)
                else:
                    cam_frame = None

            if cam_frame is not None:
                h, w = cam_frame.shape[:2]
            else:
                h, w = 480, 640
            left = reference_panel(
                choreo, pose, t, clock.playing, w, h, cfg.confidence, cfg.mirror
            )

            if cam_frame is not None:
                right = cam_frame
                if live is not None:
                    draw_pose(right, live.keypoints, cfg.confidence)
                _draw_text(
                    right,
                    [
                        ("YOU — live camera", TEXT, 2),
                        (f"FPS {fps.tick():4.1f}", TEXT, 1),
                    ],
                )
            else:
                right = np.full((h, w, 3), PANEL_BG, dtype=np.uint8)
                _draw_text(
                    right,
                    [
                        ("NO CAMERA", WARN, 2),
                        ("reference playback still works", TEXT, 1),
                        ("camera is needed for practice mode", TEXT, 1),
                    ],
                )
            _draw_text(
                right,
                [("SPACE pause | arrows seek | R restart | ENTER practice | Q quit", TEXT, 1)],
                y=h - 30,
            )

            cv2.imshow(WINDOW_NAME, np.hstack([left, right]))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (13, 10):  # ENTER
                start_practice = True
                break
            if key == 32:  # SPACE
                clock.toggle()
            elif key in (ord("r"), ord("R")):
                clock.restart()
            elif key == 81 or key == 2:  # left arrow (backend-dependent)
                clock.skip(-SEEK_STEP)
            elif key == 83 or key == 3:  # right arrow
                clock.skip(SEEK_STEP)
    finally:
        cv2.destroyAllWindows()

    if not start_practice:
        if cap is not None:
            cap.release()
        return 0
    if cap is None:
        print("Error: practice mode needs a camera — connect one and relaunch.")
        return 2
    return run_practice_loop(cfg, tracker, choreo, moves, cap, device)
