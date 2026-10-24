"""Camera capture wrapper for the live pose tracker."""

from __future__ import annotations

import cv2

from .config import AppConfig


class CameraError(RuntimeError):
    pass


def open_camera(cfg: AppConfig) -> cv2.VideoCapture:
    """Open the configured camera, or raise CameraError with guidance."""
    cap = cv2.VideoCapture(cfg.camera_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cfg.camera_index)
    if not cap.isOpened():
        cap.release()
        raise CameraError(
            f"Cannot open camera index {cfg.camera_index}. "
            "Check that a webcam is connected and not in use by another app, "
            "or try a different index with --camera."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_height)
    # Keep the driver buffer shallow so we process fresh frames rather than
    # draining a backlog of stale ones.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap
