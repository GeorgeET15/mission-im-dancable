"""Reference choreography extraction and caching.

Extracts per-frame pose keypoints from a local dance video and caches them
so practice mode does not re-run inference on every launch.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .pose import NUM_KEYPOINTS, PoseTracker

CACHE_VERSION = 1


class ReferenceVideoError(RuntimeError):
    pass


@dataclass
class ReferenceChoreography:
    """Pose landmarks sampled from a reference dance video.

    keypoints has shape (N, NUM_KEYPOINTS, 3): x, y pixel coords and
    confidence. Samples where no person was detected are all zeros.
    """

    video_path: Path
    video_fps: float
    frame_count: int
    width: int
    height: int
    sample_fps: float
    timestamps: np.ndarray  # (N,) seconds
    keypoints: np.ndarray  # (N, NUM_KEYPOINTS, 3)
    model_name: str
    video_size: int = 0
    video_mtime_ns: int = 0

    @property
    def duration(self) -> float:
        return self.frame_count / self.video_fps if self.video_fps else 0.0

    @property
    def num_samples(self) -> int:
        return int(self.timestamps.shape[0])

    def detected_count(self) -> int:
        """Number of samples where a person was detected."""
        return int((self.keypoints[:, :, 2].max(axis=1) > 0).sum())


def window_indices(choreo: ReferenceChoreography, t: float, window_s: float) -> np.ndarray:
    """Sample indices whose timestamps fall within t ± window_s."""
    ts = choreo.timestamps
    lo = np.searchsorted(ts, t - window_s)
    hi = np.searchsorted(ts, t + window_s, side="right")
    return np.arange(lo, hi)


def nearest_index(choreo: ReferenceChoreography, t: float) -> int | None:
    """Index of the sample closest to t, or None for an empty choreography."""
    ts = choreo.timestamps
    if ts.size == 0:
        return None
    i = int(np.searchsorted(ts, t))
    candidates = [j for j in (i - 1, i) if 0 <= j < ts.size]
    if not candidates:
        return None
    return min(candidates, key=lambda j: abs(ts[j] - t))


def sampled_frame_indices(frame_count: int, video_fps: float, sample_fps: float) -> list[int]:
    """Frame indices to run inference on for the requested sample rate."""
    if frame_count <= 0 or video_fps <= 0 or sample_fps <= 0:
        return []
    step = max(1, round(video_fps / sample_fps))
    return list(range(0, frame_count, step))


def extract_reference(
    tracker: PoseTracker,
    video_path: Path,
    sample_fps: float,
    progress_every: int = 25,
) -> ReferenceChoreography:
    """Run pose inference over a video at the requested sample rate."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise ReferenceVideoError(
            f"Cannot open video {video_path}. Check the path and that the format is supported."
        )
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    wanted = set(sampled_frame_indices(frame_count, video_fps, sample_fps))
    timestamps: list[float] = []
    keypoints: list[np.ndarray] = []
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx in wanted:
                result = tracker.predict(frame)
                timestamps.append(idx / video_fps)
                keypoints.append(
                    result.keypoints
                    if result is not None
                    else np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)
                )
                if progress_every and len(timestamps) % progress_every == 0:
                    print(
                        f"  extracted {len(timestamps)}/{len(wanted)} pose samples...",
                        flush=True,
                    )
            idx += 1
    finally:
        cap.release()

    if not timestamps:
        raise ReferenceVideoError(
            f"No frames decoded from {video_path} — the file may be corrupt or unsupported."
        )

    stat = video_path.stat()
    return ReferenceChoreography(
        video_path=video_path,
        video_fps=video_fps,
        frame_count=frame_count,
        width=width,
        height=height,
        sample_fps=sample_fps,
        timestamps=np.asarray(timestamps, dtype=np.float64),
        keypoints=np.asarray(keypoints, dtype=np.float32),
        model_name="",
        video_size=stat.st_size,
        video_mtime_ns=stat.st_mtime_ns,
    )


def _cache_key(video_path: Path) -> str:
    return hashlib.sha1(str(video_path.resolve()).encode()).hexdigest()[:16]


def cache_path_for(video_path: Path, cache_dir: Path) -> Path:
    return cache_dir / f"{video_path.stem}.{_cache_key(video_path)}.npz"


def _matches_meta(
    meta: dict,
    video_path: Path,
    sample_fps: float,
    model_name: str,
    confidence: float,
) -> bool:
    """Staleness check: video file, extraction settings, and model must match."""
    try:
        stat = video_path.stat()
    except OSError:
        return False
    return (
        meta.get("cache_version") == CACHE_VERSION
        and meta.get("video_size") == stat.st_size
        and meta.get("video_mtime_ns") == stat.st_mtime_ns
        and meta.get("sample_fps") == sample_fps
        and meta.get("model_name") == model_name
        and meta.get("confidence") == confidence
    )


def save_cache(choreo: ReferenceChoreography, cache_path: Path, model_name: str, confidence: float) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "cache_version": CACHE_VERSION,
        "video_path": str(choreo.video_path.resolve()),
        "video_fps": choreo.video_fps,
        "frame_count": choreo.frame_count,
        "width": choreo.width,
        "height": choreo.height,
        "sample_fps": choreo.sample_fps,
        "video_size": choreo.video_size,
        "video_mtime_ns": choreo.video_mtime_ns,
        "model_name": model_name,
        "confidence": confidence,
    }
    np.savez(
        cache_path,
        meta=json.dumps(meta),
        timestamps=choreo.timestamps,
        keypoints=choreo.keypoints,
    )


def load_cache(cache_path: Path, video_path: Path, sample_fps: float, model_name: str, confidence: float) -> ReferenceChoreography | None:
    """Load a cached choreography, or None when missing, corrupt, or stale."""
    if not cache_path.is_file():
        return None
    try:
        data = np.load(cache_path, allow_pickle=False)
        meta = json.loads(str(data["meta"]))
        choreo = ReferenceChoreography(
            video_path=video_path,
            video_fps=float(meta["video_fps"]),
            frame_count=int(meta["frame_count"]),
            width=int(meta["width"]),
            height=int(meta["height"]),
            sample_fps=float(meta["sample_fps"]),
            timestamps=np.asarray(data["timestamps"], dtype=np.float64),
            keypoints=np.asarray(data["keypoints"], dtype=np.float32),
            model_name=str(meta["model_name"]),
            video_size=int(meta["video_size"]),
            video_mtime_ns=int(meta["video_mtime_ns"]),
        )
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return None
    if not _matches_meta(meta, video_path, sample_fps, model_name, confidence):
        return None
    return choreo


def load_or_extract(
    tracker: PoseTracker,
    video_path: Path,
    sample_fps: float,
    cache_dir: Path,
    confidence: float,
    model_name: str,
) -> ReferenceChoreography:
    """Return a cached extraction when fresh, otherwise extract and cache."""
    cache_path = cache_path_for(video_path, cache_dir)
    cached = load_cache(cache_path, video_path, sample_fps, model_name, confidence)
    if cached is not None:
        print(f"Loaded cached reference poses from {cache_path}")
        return cached

    start = time.perf_counter()
    print(f"Extracting reference poses from {video_path} at {sample_fps:g} fps...")
    choreo = extract_reference(tracker, video_path, sample_fps)
    choreo.model_name = model_name
    elapsed = time.perf_counter() - start
    print(
        f"Extracted {choreo.num_samples} samples ({choreo.detected_count()} with a "
        f"person detected) in {elapsed:.1f}s"
    )
    if choreo.detected_count() == 0:
        print(
            "Warning: no person was detected in any sampled frame. "
            "Practice mode will have no reference poses to compare against."
        )
    save_cache(choreo, cache_path, model_name, confidence)
    print(f"Cached reference poses to {cache_path}")
    return choreo
