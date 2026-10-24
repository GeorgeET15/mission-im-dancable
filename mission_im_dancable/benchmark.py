"""Pose-backend benchmark: decode once, run each backend on identical frames.

YOLO stays the default backend — this module is for comparison only.
Results land in an ignored output directory as JSON + a text summary.
"""

from __future__ import annotations

import hashlib
import json
import platform
import resource
import statistics
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from .geometry import normalize_keypoints
from .mediapipe_pose import (
    MediaPipeUnavailableError,
    model_path_for,
)
from .pose import PoseTracker

OUTPUT_DIR = Path("output/pose_benchmark")
FPS_DEFAULT = 30.0
YOLO_MODEL_DIR = Path("models")


def yolo_model_path(variant: str, model_dir: Path = YOLO_MODEL_DIR) -> Path:
    """Weights path for a `yolo-<variant>` backend (e.g. 11m → yolo11m-pose.pt).

    The bare `yolo` backend is always the default yolo11n-pose.pt. Variants
    are never silently substituted — a missing file fails loudly.
    """
    return model_dir / f"yolo{variant}-pose.pt"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_video_meta(video_path: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"cannot open video {video_path}")
    meta = {
        "path": str(video_path.resolve()),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or FPS_DEFAULT),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return meta


def iter_frames(video_path: Path, limit: int | None = None):
    """Yield (index, BGR frame) one at a time; releases the capture.

    Streaming decode keeps memory flat — a previous version buffered the
    whole clip (~1 GB for 354 720x1280 frames) and caused OOM.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"cannot open video {video_path}")
    idx = 0
    try:
        while limit is None or idx < limit:
            ok, frame = cap.read()
            if not ok:
                break
            yield idx, frame
            idx += 1
        if idx == 0:
            raise RuntimeError(f"no frames decoded from {video_path}")
    finally:
        cap.release()


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def latency_stats(times_ms: list[float]) -> dict:
    if not times_ms:
        return {"count": 0, "mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0, "fps": 0.0}
    mean = statistics.fmean(times_ms)
    return {
        "count": len(times_ms),
        "mean_ms": round(mean, 3),
        "median_ms": round(statistics.median(times_ms), 3),
        "p95_ms": round(percentile(times_ms, 0.95), 3),
        "fps": round(1000.0 / mean, 2) if mean else 0.0,
    }


def missing_runs(detected: list[bool]) -> dict:
    """Longest consecutive missing-pose run and total misses."""
    longest = run = 0
    for d in detected:
        run = 0 if d else run + 1
        longest = max(longest, run)
    misses = sum(1 for d in detected if not d)
    return {"missing_frames": misses, "longest_missing_streak": longest}


def temporal_jitter(keypoint_seq: list[np.ndarray | None], conf_threshold: float) -> dict:
    """Median per-joint frame-to-frame displacement after torso normalization.

    Only consecutive frames where the joint was reliable in BOTH samples
    count; the raw sequence is not smoothed beforehand.
    """
    displacements: dict[int, list[float]] = {}
    for prev, cur in zip(keypoint_seq, keypoint_seq[1:]):
        if prev is None or cur is None:
            continue
        ok = (prev[:, 2] >= conf_threshold) & (cur[:, 2] >= conf_threshold)
        if ok.sum() < 6:
            continue
        pn, cn = normalize_keypoints(prev), normalize_keypoints(cur)
        for i in np.where(ok)[0]:
            displacements.setdefault(int(i), []).append(
                float(np.linalg.norm(cn[i, :2] - pn[i, :2]))
            )
    flat = [d for ds in displacements.values() for d in ds]
    return {
        "median_jitter": round(statistics.median(flat), 4) if flat else 0.0,
        "mean_jitter": round(statistics.fmean(flat), 4) if flat else 0.0,
        "joints_evaluated": len(displacements),
        "transition_count": len(flat),
        "note": "includes real motion — lower is not automatically better",
    }


def world_jitter(world_seq: list[np.ndarray], conf_threshold: float) -> dict:
    """Frame-to-frame displacement of world landmarks (metres, hip-centred).

    World landmarks are already model-relative 3D estimates, so no torso
    normalization is needed. Counts only joints visible in both frames.
    """
    displacements: list[float] = []
    vis_sum = vis_n = 0.0
    for prev, cur in zip(world_seq, world_seq[1:]):
        ok = (prev[:, 3] >= conf_threshold) & (cur[:, 3] >= conf_threshold)
        if ok.sum() < 6:
            continue
        displacements.extend(
            np.linalg.norm(cur[ok, :3] - prev[ok, :3], axis=1).tolist()
        )
    for w in world_seq:
        vis_sum += float(w[:, 3].mean())
        vis_n += 1
    return {
        "median_disp_m": round(statistics.median(displacements), 4) if displacements else 0.0,
        "mean_disp_m": round(statistics.fmean(displacements), 4) if displacements else 0.0,
        "mean_visibility": round(vis_sum / vis_n, 3) if vis_n else 0.0,
        "transition_count": len(displacements),
        "note": "includes real motion; world coords are model estimates, not mocap",
    }


def _env_info() -> dict:
    info = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["torch"] = None
    try:
        import mediapipe as mp
        info["mediapipe"] = mp.__version__
    except ImportError:
        info["mediapipe"] = None
    try:
        import ultralytics
        info["ultralytics"] = ultralytics.__version__
    except ImportError:
        info["ultralytics"] = None
    info["opencv"] = cv2.__version__
    info["numpy"] = np.__version__
    return info


def _gpu_peak_mb(device: str) -> float | None:
    """Peak CUDA allocation for this process, or None without torch/CUDA."""
    if device != "cuda":
        return None
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 2**20, 1)
    except Exception:
        pass
    return None


def run_benchmark(
    video_path: Path,
    backend: str,
    confidence: float,
    device: str = "cuda",
    model_path: Path | None = None,
    fps: float = FPS_DEFAULT,
    warmup: int = 5,
    limit: int | None = None,
    visualize_path: Path | None = None,
) -> dict:
    """Stream `video_path` through one backend; bounded memory throughout.

    Frames are decoded one at a time and dropped after use; only small
    per-frame metrics (detected flag, 17x3 keypoints, 33x4 world) are
    retained. When `visualize_path` is set, each annotated frame is
    written to the VideoWriter immediately — nothing accumulates.
    """
    spec = {"backend": backend, "device": device}
    if backend == "yolo" or backend.startswith("yolo-"):
        if backend == "yolo":
            ypath = Path(model_path) if model_path else YOLO_MODEL_DIR / "yolo11n-pose.pt"
        else:
            # Explicit --model wins; otherwise resolve the variant filename.
            ypath = Path(model_path) if model_path else yolo_model_path(
                backend.split("-", 1)[1], YOLO_MODEL_DIR)
        t0 = time.perf_counter()
        tracker = PoseTracker(ypath, device, confidence)
        init_s = time.perf_counter() - t0
        infer = tracker.predict
        spec["model_path"] = str(ypath)
        spec["model_file"] = ypath.name
        spec["model_sha256"] = sha256_of(ypath)
        spec["device_verified"] = None  # set after warm-up (params move lazily)
    elif backend.startswith("mediapipe"):
        parts = backend.split("-")[1:]
        delegate = "gpu" if "gpu" in parts else "cpu"
        variant = next((p for p in parts if p != "gpu"), "heavy")
        mp_path = model_path or model_path_for(variant)
        from .mediapipe_pose import MediaPipePoseTracker

        t0 = time.perf_counter()
        tracker = MediaPipePoseTracker(mp_path, confidence, confidence, delegate)
        init_s = time.perf_counter() - t0
        infer = lambda f: tracker.track(f, fps)
        spec["variant"] = variant
        spec["model_path"] = str(mp_path)
        spec["delegate"] = delegate
    else:
        raise ValueError(f"unknown backend {backend!r}")

    detected: list[bool] = []
    missing_idx: list[int] = []
    kpts_seq: list[np.ndarray | None] = []
    world_samples: list[np.ndarray] = []
    infer_ms, e2e_ms = [], []
    writer = None
    # Measure inference-phase VRAM, not the load spike.
    try:
        if str(device).startswith("cuda"):
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass
    wall0, cpu0 = time.perf_counter(), time.process_time()
    try:
        for idx, frame in iter_frames(video_path, limit):
            if idx < warmup:  # warm-up: runs inference, excluded from stats
                infer(frame)
                if idx == warmup - 1 and backend.startswith("yolo"):
                    try:  # inference copy lives under predictor.model (AutoBackend)
                        import torch
                        pred = tracker._model.predictor
                        inner = getattr(pred.model, "model", None)
                        spec["device_verified"] = bool(
                            str(device).startswith("cuda")
                            and torch.cuda.is_available()
                            and str(pred.device).startswith("cuda")
                            and inner is not None
                            and any(p.is_cuda for p in inner.parameters()))
                    except Exception:
                        spec["device_verified"] = False
                continue
            t0 = time.perf_counter()
            result = infer(frame)
            e2e = (time.perf_counter() - t0) * 1000.0
            e2e_ms.append(e2e)
            # Missed detections still ran inference; the wrapper does not
            # expose their inner timing, so fall back to the call duration.
            infer_ms.append(result.inference_ms if result else e2e)
            detected.append(result is not None)
            if result is None:
                missing_idx.append(idx)
            kpts_seq.append(result.keypoints if result else None)
            if result is not None:
                world = getattr(result, "world_landmarks", None)
                if world is not None:
                    world_samples.append(world)
            if visualize_path is not None:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(visualize_path), cv2.VideoWriter_fourcc(*"mp4v"),
                        fps, (w, h))
                from .render import draw_pose

                if result is not None:
                    draw_pose(frame, result.keypoints, confidence)
                cv2.putText(frame,
                            f"{backend} | {e2e:.1f} ms | {'POSE' if result else 'NO POSE'}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 255), 2)
                writer.write(frame)
            del frame, result
    finally:
        if writer is not None:
            writer.release()
        if hasattr(tracker, "close"):
            tracker.close()
    wall_s = time.perf_counter() - wall0
    cpu_s = time.process_time() - cpu0

    scores = [k[k[:, 2] > 0, 2].mean() for k in kpts_seq
              if k is not None and (k[:, 2] > 0).any()]
    result = {
        "backend": spec,
        "init_seconds": round(init_s, 3),
        "warmup_frames": warmup,
        "frames_evaluated": len(detected),
        "inference": latency_stats(infer_ms),
        "end_to_end": latency_stats(e2e_ms),
        "reliability": {
            "detection_rate": round(sum(detected) / len(detected), 4) if detected else 0.0,
            **missing_runs(detected),
            "missing_frame_indices": missing_idx[:100],
            "mean_landmark_score": round(float(np.mean(scores)), 3) if scores else 0.0,
        },
        "jitter": temporal_jitter(kpts_seq, confidence),
        "world_landmark_frames": len(world_samples),
        "resources": {
            "wall_s": round(wall_s, 2),
            "cpu_s": round(cpu_s, 2),
            "peak_rss_mb": round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            "gpu_peak_mb": _gpu_peak_mb(
                device if backend.startswith("yolo") else "cpu"),
        },
    }
    if visualize_path is not None:
        result["visualization"] = str(visualize_path)
    if world_samples:
        result["world"] = world_jitter(world_samples, confidence)
        result["_world_seq"] = np.stack(world_samples)
    return result


def summarize(results: list[dict]) -> str:
    lines = ["backend | init s | mean ms | median | p95 | fps | det% | longest miss | jitter med"]
    for r in results:
        if "error" in r:
            lines.append(f"{r['backend'].get('backend', '?')} | ERROR | "
                         f"{r['error'][:60]}")
            continue
        b, i, rel, j = r["backend"], r["inference"], r["reliability"], r["jitter"]
        lines.append(
            f"{b['backend']} | {r['init_seconds']} | {i['mean_ms']} | {i['median_ms']} | "
            f"{i['p95_ms']} | {i['fps']} | {rel['detection_rate']} | "
            f"{rel['longest_missing_streak']} | {j['median_jitter']}"
        )
    return "\n".join(lines)


def save_results(results: list[dict], meta: dict, out_dir: Path = OUTPUT_DIR) -> Path:
    """Merge per-backend results into pose_benchmark.json + summary.txt.

    Results are keyed by backend name so sequential one-backend-at-a-time
    runs accumulate instead of overwriting each other.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "pose_benchmark.json"
    merged: dict[str, dict] = {}
    if path.is_file():
        try:
            for r in json.loads(path.read_text()).get("results", []):
                merged[r["backend"]["backend"]] = r
        except (json.JSONDecodeError, KeyError):
            pass  # corrupt/old-format file: start fresh
    for r in results:
        slim = {k: v for k, v in r.items()
                if k != "records" and not k.startswith("_")}
        merged[slim["backend"]["backend"]] = slim
    ordered = [merged[k] for k in dict.fromkeys(
        [r["backend"]["backend"] for r in results] + sorted(merged))]
    payload = {"video": meta, "environment": _env_info(), "results": ordered}
    path.write_text(json.dumps(payload, indent=2))
    (out_dir / "summary.txt").write_text(summarize(ordered))
    return path


def nvidia_gpu_stats() -> dict:
    """Best-effort nvidia-smi snapshot; empty dict when unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            name, util, mem = [p.strip() for p in out.stdout.split(",")]
            return {"gpu": name, "utilization": util, "memory_used": mem}
    except (OSError, subprocess.SubprocessError):
        pass
    return {}
