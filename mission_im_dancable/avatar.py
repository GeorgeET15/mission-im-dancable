"""Offline 2D-reference to Amy-rig render, cached per choreography."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .reference import ReferenceChoreography

RIG_VERSION = 1
MODEL_PATH = Path(__file__).resolve().parent.parent / "assets/blender/mission_dancer.blend"
BONES = (
    "Spine2", "Neck",
    "LeftArm", "LeftForeArm", "RightArm", "RightForeArm",
    "LeftUpLeg", "LeftLeg", "RightUpLeg", "RightLeg",
)
SEGMENTS = (
    (None, None), (None, 0),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 13), (13, 15), (12, 14), (14, 16),
)


def pose_targets(
    keypoints: np.ndarray, confidence: float = 0.35,
) -> np.ndarray:
    """Unit bone directions in the camera-facing XZ plane; NaN if unavailable.

    This preserves visible limb directions without claiming monocular depth.
    """
    out = np.full((len(BONES), 3), np.nan, dtype=np.float32)
    if keypoints.shape != (17, 3):
        raise ValueError("Expected COCO17 keypoints (17, 3)")

    def midpoint(a: int, b: int) -> np.ndarray | None:
        if min(keypoints[a, 2], keypoints[b, 2]) < confidence:
            return None
        return (keypoints[a, :2] + keypoints[b, :2]) / 2

    hips = midpoint(11, 12)
    shoulders = midpoint(5, 6)
    for i, (start, end) in enumerate(SEGMENTS):
        if i == 0:
            a, b = hips, shoulders
        elif i == 1:
            a = shoulders
            b = keypoints[0, :2] if keypoints[0, 2] >= confidence else None
        else:
            a = keypoints[start, :2] if keypoints[start, 2] >= confidence else None
            b = keypoints[end, :2] if keypoints[end, 2] >= confidence else None
        if a is None or b is None:
            continue
        dx, dy = b - a
        vec = np.array([dx, 0.0, -dy], dtype=np.float32)
        norm = np.linalg.norm(vec)
        if np.isfinite(norm) and norm > 1e-4:
            out[i] = vec / norm
    return out


def animation_targets(choreo: ReferenceChoreography, confidence: float) -> tuple[np.ndarray, np.ndarray]:
    """Return per-sample bone targets and root translations (Blender cm).

    A missing landmark leaves its bone at rest, rather than treating (0, 0)
    as a measured joint. Translation is anchored to the first valid hip.
    """
    targets = np.stack([pose_targets(p, confidence) for p in choreo.keypoints])
    roots = np.zeros((choreo.num_samples, 2), dtype=np.float32)
    origin = None
    for i, pose in enumerate(choreo.keypoints):
        if min(pose[11, 2], pose[12, 2], pose[5, 2], pose[6, 2]) < confidence:
            continue
        hip = (pose[11, :2] + pose[12, :2]) / 2
        shoulder = (pose[5, :2] + pose[6, :2]) / 2
        torso = np.linalg.norm(shoulder - hip)
        if torso < 1e-4:
            continue
        if origin is None:
            origin = hip.copy()
        roots[i] = np.clip((hip - origin) / torso * np.array([18, -18]), -20, 20)
    return targets, roots


def cache_path(
    choreo: ReferenceChoreography, cache_dir: Path,
    model: Path = MODEL_PATH, confidence: float = 0.35,
) -> Path:
    stat = model.stat()
    fingerprint = hashlib.sha256()
    fingerprint.update(str(choreo.video_path.resolve()).encode())
    fingerprint.update(choreo.timestamps.tobytes())
    fingerprint.update(choreo.keypoints.tobytes())
    fingerprint.update(f"{stat.st_size}:{stat.st_mtime_ns}:{confidence}:{RIG_VERSION}".encode())
    return cache_dir / "amy" / f"{choreo.video_path.stem}-{fingerprint.hexdigest()[:16]}"


def frame_manifest(directory: Path, sample_count: int) -> dict | None:
    """Reject incomplete or corrupt render caches."""
    try:
        info = json.loads((directory / "manifest.json").read_text())
        if info["version"] != RIG_VERSION or len(info["timestamps"]) != sample_count:
            return None
        if any(not (directory / f"frame_{i:04d}.png").is_file() for i in range(1, sample_count + 1)):
            return None
        return info
    except (OSError, KeyError, ValueError, TypeError):
        return None


def render_choreography(
    choreo: ReferenceChoreography,
    cache_dir: Path,
    confidence: float = 0.35,
    on_status=None,
    model: Path = MODEL_PATH,
    blender: str | None = None,
) -> Path:
    """Render reference-driven Amy PNGs offline; return a complete cache dir.

    Raises RuntimeError on unavailable Blender or failed render. The caller
    should fall back to the extracted skeleton, never to the canned sway.
    """
    if choreo.num_samples == 0 or not model.is_file():
        raise RuntimeError("Amy model or reference samples unavailable")
    base = cache_path(choreo, cache_dir, model, confidence)
    for attempt in range(10):
        path = base if attempt == 0 else Path(f"{base}-retry{attempt}")
        if frame_manifest(path, choreo.num_samples) is not None:
            return path
        if not path.exists():
            break
    else:
        raise RuntimeError("Too many incomplete Amy caches; showing reference skeleton")
    blender = blender or shutil.which("blender")
    if blender is None:
        raise RuntimeError("Blender unavailable; showing reference skeleton")
    targets, roots = animation_targets(choreo, confidence)
    if np.isfinite(targets).all(axis=2).sum() == 0:
        raise RuntimeError("No reliable reference joints to animate Amy")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="amy-render-", dir=path.parent) as tmp:
        work = Path(tmp)
        np.savez(work / "targets.npz", targets=targets, roots=roots)
        if on_status:
            on_status(f"Rendering Amy to {choreo.num_samples} reference poses...")
        cmd = [blender, "-b", str(model), "--python",
               str(Path(__file__).with_name("avatar_render.py")), "--",
               str(work / "targets.npz"), str(work)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"Amy render failed: {exc}") from exc
        if result.returncode != 0:
            raise RuntimeError(f"Amy render failed: {result.stderr[-600:]}")
        timestamps = choreo.timestamps.tolist()
        (work / "manifest.json").write_text(
            json.dumps({"version": RIG_VERSION, "timestamps": timestamps})
        )
        if frame_manifest(work, choreo.num_samples) is None:
            raise RuntimeError("Amy render incomplete; showing reference skeleton")
        work.rename(path)
    return path
