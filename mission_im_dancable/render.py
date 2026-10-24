"""OpenCV drawing helpers for the live pose tracker."""

from __future__ import annotations

import cv2
import numpy as np

from .geometry import SKELETON_EDGES

SKELETON_COLOR = (0, 220, 180)  # BGR — mission-control teal
KEYPOINT_COLOR = (60, 60, 255)  # BGR — red-orange joints
HUD_COLOR = (220, 220, 220)


def draw_pose(frame: np.ndarray, keypoints: np.ndarray, conf_threshold: float) -> np.ndarray:
    """Draw skeleton edges and keypoint joints onto the frame in place."""
    for a, b in SKELETON_EDGES:
        pa, pb = keypoints[a], keypoints[b]
        if pa[2] >= conf_threshold and pb[2] >= conf_threshold:
            cv2.line(
                frame,
                (int(pa[0]), int(pa[1])),
                (int(pb[0]), int(pb[1])),
                SKELETON_COLOR,
                2,
                cv2.LINE_AA,
            )
    for x, y, conf in keypoints:
        if conf >= conf_threshold:
            cv2.circle(frame, (int(x), int(y)), 4, KEYPOINT_COLOR, -1, cv2.LINE_AA)
    return frame


def fit_keypoints(
    keypoints: np.ndarray,
    src_w: int,
    src_h: int,
    dst_x: float,
    dst_y: float,
    dst_w: float,
    dst_h: float,
    fill: float = 0.9,
) -> np.ndarray:
    """Scale keypoints from source frame coords into a destination rect.

    Uniform scaling centred in the rect; confidence column is untouched.
    """
    out = keypoints.copy()
    if src_w <= 0 or src_h <= 0:
        return out
    scale = min(dst_w / src_w, dst_h / src_h) * fill
    out[:, 0] = out[:, 0] * scale + dst_x + (dst_w - src_w * scale) / 2
    out[:, 1] = out[:, 1] * scale + dst_y + (dst_h - src_h * scale) / 2
    return out


def draw_hud(
    frame: np.ndarray,
    fps: float,
    inference_ms: float | None,
    device: str,
    person_detected: bool,
) -> np.ndarray:
    """Draw the FPS / inference / status overlay in place."""
    infer_text = f"{inference_ms:.1f} ms" if inference_ms is not None else "n/a"
    status = "PERSON DETECTED" if person_detected else "NO PERSON"
    lines = [
        "MISSION IM-DANCABLE",
        f"FPS {fps:5.1f}  |  infer {infer_text}  |  {device.upper()}",
        status,
        "Q / Esc to quit",
    ]
    for i, text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (10, 24 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            HUD_COLOR if i != 2 else SKELETON_COLOR,
            1 if i != 0 else 2,
            cv2.LINE_AA,
        )
    return frame
