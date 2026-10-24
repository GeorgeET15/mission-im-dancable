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
    # Practice mode (active when source is set; camera mode otherwise).
    source: Path | None = None
    moves_path: Path | None = None
    sample_fps: float = 10.0
    window_s: float = 0.15
    mirror: bool = False
    cache_dir: Path = Path("cache")
    game: bool = False  # pygame rhythm-game mode instead of OpenCV practice
    preview: bool = False  # animated reference playback before practice
    no_cam: bool = False  # preview without opening the camera
    benchmark_pose: bool = False  # benchmark pose backends on --source
    visualize_pose: bool = False  # render per-backend skeleton overlay videos
    pose_backends: str = "yolo,mediapipe-heavy"  # comma-separated list
    max_frames: int = 30  # benchmark frame cap incl. warmup; 0 = whole video


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
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        metavar="VIDEO",
        help="reference choreography video; enables practice mode",
    )
    parser.add_argument(
        "--moves",
        type=Path,
        default=None,
        metavar="JSON",
        help="move segmentation file for the reference video",
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=10.0,
        help="pose extraction rate for the reference video (default: 10)",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=0.15,
        metavar="SECONDS",
        help="match window half-width around the reference pose (default: 0.15)",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="mirror the reference so it matches selfie-view movement",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("cache"),
        help="directory for extracted-pose caches (default: cache/)",
    )
    parser.add_argument(
        "--game",
        action="store_true",
        help="rhythm-game arcade UI (pygame); picks the video in the app",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="play the animated reference skeleton first; ENTER continues to practice",
    )
    parser.add_argument(
        "--no-cam",
        action="store_true",
        help="preview the reference without opening the camera",
    )
    parser.add_argument(
        "--benchmark-pose",
        action="store_true",
        help="benchmark pose backends on --source frames (YOLO vs MediaPipe)",
    )
    parser.add_argument(
        "--visualize-pose",
        action="store_true",
        help="with --benchmark-pose, render per-backend skeleton overlay videos",
    )
    parser.add_argument(
        "--pose-backends",
        default="yolo,mediapipe-heavy",
        metavar="LIST",
        help="comma-separated backends for --benchmark-pose (default: yolo,mediapipe-heavy)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=30,
        metavar="N",
        help="frames decoded per benchmark backend incl. warmup (default: 30; 0 = whole video)",
    )
    args = parser.parse_args(argv)

    if args.width <= 0 or args.height <= 0:
        parser.error("frame dimensions must be positive")
    if not 0.0 <= args.conf <= 1.0:
        parser.error("--conf must be between 0 and 1")
    if args.sample_fps <= 0:
        parser.error("--sample-fps must be positive")
    if args.window < 0:
        parser.error("--window must be >= 0")
    if args.moves and not args.source:
        parser.error("--moves requires --source")
    if args.preview and not args.source:
        parser.error("--preview requires --source VIDEO")
    if args.preview and args.game:
        parser.error("--preview cannot be combined with --game")
    if args.no_cam and not args.preview:
        parser.error("--no-cam only applies to --preview")
    if args.benchmark_pose and not args.source:
        parser.error("--benchmark-pose requires --source VIDEO")
    if args.visualize_pose and not args.benchmark_pose:
        parser.error("--visualize-pose requires --benchmark-pose")
    if args.max_frames < 0:
        parser.error("--max-frames must be >= 0")

    return AppConfig(
        camera_index=args.camera,
        frame_width=args.width,
        frame_height=args.height,
        model_path=args.model,
        device=args.device,
        confidence=args.conf,
        source=args.source,
        moves_path=args.moves,
        sample_fps=args.sample_fps,
        window_s=args.window,
        mirror=args.mirror,
        cache_dir=args.cache_dir,
        game=args.game,
        preview=args.preview,
        no_cam=args.no_cam,
        benchmark_pose=args.benchmark_pose,
        visualize_pose=args.visualize_pose,
        pose_backends=args.pose_backends,
        max_frames=args.max_frames,
    )
