"""Runtime configuration for the live pose tracker."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL_PATH = Path("models/yolo11n-pose.pt")


@dataclass(frozen=True)
class AppConfig:
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    model_path: Path = DEFAULT_MODEL_PATH
    device: str = "auto"  # "auto", "cuda", or "cpu"
    confidence: float = 0.3


def resolve_device(requested: str, cuda_available: bool) -> str:
    """Map a requested device string to the runtime device.

    Raises ValueError for unknown requests so typos fail loudly.
    """
    if requested == "auto":
        return "cuda" if cuda_available else "cpu"
    if requested in {"cuda", "cpu"}:
        return requested
    raise ValueError(f"Unknown device {requested!r}; expected auto, cuda, or cpu")


def parse_args(argv: list[str] | None = None) -> AppConfig:
    parser = argparse.ArgumentParser(
        prog="mission_im_dancable",
        description="Live pose tracker for Mission Im-Dancable.",
    )
    parser.add_argument("--camera", type=int, default=0, help="camera index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="frame width in pixels")
    parser.add_argument("--height", type=int, default=720, help="frame height in pixels")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="path to the YOLO pose model weights",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="inference device; auto prefers CUDA when available",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.3,
        help="keypoint confidence threshold for display (0-1)",
    )
    args = parser.parse_args(argv)

    if args.width <= 0 or args.height <= 0:
        parser.error("frame dimensions must be positive")
    if not 0.0 <= args.conf <= 1.0:
        parser.error("--conf must be between 0 and 1")

    return AppConfig(
        camera_index=args.camera,
        frame_width=args.width,
        frame_height=args.height,
        model_path=args.model,
        device=args.device,
        confidence=args.conf,
    )
