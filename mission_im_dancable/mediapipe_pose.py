"""MediaPipe Pose Landmarker adapter — benchmark-only, optional dependency.

Produces PoseResult-compatible keypoints (17 COCO rows) plus optional
world landmarks for future retargeting analysis. Not the default backend:
YOLO remains primary.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .pose import NUM_KEYPOINTS, PoseResult

MODEL_DIR = Path("models/mediapipe")
MODEL_URLS = {
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
}

# MediaPipe 33-landmark index → COCO-17 index used by this project.
# COCO: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles.
MP_TO_COCO = {
    0: 0,          # nose
    2: 1,          # left_eye (eye center)
    5: 2,          # right_eye
    7: 3,          # left_ear
    8: 4,          # right_ear
    11: 5,         # left_shoulder
    12: 6,         # right_shoulder
    13: 7,         # left_elbow
    14: 8,         # right_elbow
    15: 9,         # left_wrist
    16: 10,        # right_wrist
    23: 11,        # left_hip
    24: 12,        # right_hip
    25: 13,        # left_knee
    26: 14,        # right_knee
    27: 15,        # left_ankle
    28: 16,        # right_ankle
}


class MediaPipeUnavailableError(RuntimeError):
    """Raised when the mediapipe package or model file is missing."""


def _import():
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import vision
    except ImportError as exc:
        raise MediaPipeUnavailableError(
            "mediapipe is not installed in this interpreter. Install it in a "
            "separate environment or skip mediapipe backends."
        ) from exc
    return mp, vision


def landmarks_to_coco(
    landmarks,
    width: int,
    height: int,
) -> np.ndarray:
    """Map a 33-landmark MediaPipe result (normalized x,y + visibility) to
    the project's (17, 3) COCO array of pixel x, y and confidence.

    Non-finite or out-of-frame landmarks get confidence 0 rather than
    fabricated positions.
    """
    out = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)
    for mp_idx, coco_idx in MP_TO_COCO.items():
        lm = landmarks[mp_idx]
        x, y = lm.x * width, lm.y * height
        # MediaPipe may emit normalized coords outside [0, 1] when the model
        # extrapolates a joint past the frame edge; treat them like missing.
        if (
            not (np.isfinite(x) and np.isfinite(y))
            or x < 0 or y < 0 or x >= width or y >= height
        ):
            continue
        conf = float(min(getattr(lm, "visibility", 1.0), getattr(lm, "presence", 1.0)))
        out[coco_idx] = (x, y, conf)
    return out


def world_landmarks_to_array(world_landmarks) -> np.ndarray | None:
    """(33, 4) array of world x, y, z meters + visibility for diagnostics.

    MediaPipe world landmarks are model estimates relative to the hip
    midpoint — not ground-truth motion capture.
    """
    if not world_landmarks:
        return None
    return np.asarray(
        [[lm.x, lm.y, lm.z, getattr(lm, "visibility", 0.0)] for lm in world_landmarks],
        dtype=np.float32,
    )


@dataclass
class MPResult(PoseResult):
    world_landmarks: np.ndarray | None = field(default=None)
    raw_count: int = 0


class MediaPipePoseTracker:
    """VIDEO-mode wrapper around PoseLandmarker for sequential frames.

    Timestamps must be strictly monotonically increasing; track() derives
    them from the frame index so the caller never needs a wall clock.
    """

    def __init__(
        self,
        model_path: Path | str,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        delegate: str = "cpu",
    ) -> None:
        if delegate not in {"cpu", "gpu"}:
            raise ValueError(f"unknown delegate {delegate!r}; expected cpu or gpu")
        model_path = Path(model_path)
        if not model_path.is_file():
            raise MediaPipeUnavailableError(
                f"MediaPipe model not found at {model_path} — download the "
                f".task bundle from {MODEL_URLS.get('heavy')}"
            )
        mp, vision = _import()
        t0 = time.perf_counter()
        kwargs: dict = {"model_asset_path": str(model_path)}
        if delegate == "gpu":
            # GL delegate via EGL. On this machine EGL binds the AMD iGPU
            # (Mesa/radeonsi), NOT the NVIDIA card — verify before claiming.
            kwargs["delegate"] = mp.tasks.BaseOptions.Delegate.GPU
        base = mp.tasks.BaseOptions(**kwargs)
        options = vision.PoseLandmarkerOptions(
            base_options=base,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=False,
        )
        self._mp = mp
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self.init_seconds = time.perf_counter() - t0
        self._frame_index = 0
        self._closed = False

    def _timestamp_ms(self, fps: float) -> int:
        ts = int(self._frame_index * 1000.0 / fps)
        self._frame_index += 1
        return ts

    def track(self, frame_bgr: np.ndarray, fps: float = 30.0) -> MPResult | None:
        """Process one BGR frame; returns None when no pose is found."""
        if self._closed:
            raise RuntimeError("tracker is closed")
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        start = time.perf_counter()
        result = self._landmarker.detect_for_video(image, self._timestamp_ms(fps))
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if not result.pose_landmarks:
            return None
        h, w = frame_bgr.shape[:2]
        kpts = landmarks_to_coco(result.pose_landmarks[0], w, h)
        world = world_landmarks_to_array(
            result.pose_world_landmarks[0] if result.pose_world_landmarks else None
        )
        return MPResult(
            keypoints=kpts,
            inference_ms=elapsed_ms,
            world_landmarks=world,
            raw_count=len(result.pose_landmarks),
        )

    def predict(self, frame_bgr: np.ndarray) -> MPResult | None:
        """PoseTracker-compatible alias for track() at 30 fps."""
        return self.track(frame_bgr)

    def close(self) -> None:
        if not self._closed:
            self._landmarker.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def model_path_for(variant: str, model_dir: Path = MODEL_DIR) -> Path:
    """Path for a named variant; never substitutes another variant."""
    if variant not in MODEL_URLS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {sorted(MODEL_URLS)}")
    return model_dir / f"pose_landmarker_{variant}.task"
