"""YOLO pose inference wrapper."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

NUM_KEYPOINTS = 17  # COCO pose format


@dataclass
class PoseResult:
    """Keypoints for the most confident detected person.

    keypoints has shape (NUM_KEYPOINTS, 3): x, y pixel coords and confidence.
    """

    keypoints: np.ndarray
    inference_ms: float


class PoseTracker:
    """Loads YOLO11n-Pose once and runs per-frame inference."""

    def __init__(self, model_path: Path | str, device: str, confidence: float) -> None:
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Model weights not found at {model_path}. "
                "Download yolo11n-pose.pt or pass --model PATH."
            )
        from ultralytics import YOLO

        self._model = YOLO(str(model_path))
        self._device = device
        self._confidence = confidence

    def predict(self, frame: np.ndarray) -> PoseResult | None:
        """Run pose inference on a BGR frame.

        Returns keypoints for the highest-confidence person, or None when no
        person is detected. Inference runs synchronously so frames cannot
        queue up behind the model.
        """
        import torch

        start = time.perf_counter()
        with torch.inference_mode():
            results = self._model.predict(
                frame,
                device=self._device,
                conf=self._confidence,
                verbose=False,
            )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        kpts = results[0].keypoints
        if kpts is None or kpts.xy.shape[0] == 0:
            return None

        # Highest mean-confidence person; conf may be all zeros if the model
        # did not emit scores, in which case index 0 is a fine fallback.
        mean_conf = (
            kpts.conf.mean(dim=1)
            if kpts.conf is not None
            else np.zeros(kpts.xy.shape[0])
        )
        best = int(mean_conf.argmax())
        xy = kpts.xy[best].cpu().numpy()
        conf = (
            kpts.conf[best].cpu().numpy()
            if kpts.conf is not None
            else np.ones(NUM_KEYPOINTS, dtype=np.float32)
        )
        keypoints = np.concatenate([xy, conf[:, None]], axis=1)
        return PoseResult(keypoints=keypoints, inference_ms=elapsed_ms)

    def close(self) -> None:
        """Drop the model reference so a benchmark worker can release it."""
        self._model = None
