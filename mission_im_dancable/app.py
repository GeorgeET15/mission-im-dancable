"""Live pose tracker application loop."""

from __future__ import annotations

import cv2
import torch

from .camera import CameraError, open_camera
from .config import AppConfig, resolve_device
from .pose import PoseTracker
from .render import draw_hud, draw_pose
from .timing import FpsCounter

WINDOW_NAME = "Mission Im-Dancable — live pose tracker"


def run(cfg: AppConfig) -> int:
    """Run the camera + pose loop until the user quits. Returns an exit code."""
    if cfg.source is not None:
        from .practice import run_practice

        return run_practice(cfg)

    device = resolve_device(cfg.device, torch.cuda.is_available())
    tracker = PoseTracker(cfg.model_path, device, cfg.confidence)

    try:
        cap = open_camera(cfg)
    except CameraError as exc:
        print(f"Error: {exc}")
        return 2

    fps = FpsCounter()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    print(f"Tracking on {device.upper()} — press Q or Escape in the window to quit.", flush=True)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Error: failed to read a frame from the camera.")
                return 3

            result = tracker.predict(frame)
            if result is not None:
                draw_pose(frame, result.keypoints, cfg.confidence)
            draw_hud(
                frame,
                fps.tick(),
                result.inference_ms if result else None,
                device,
                result is not None,
            )
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0
