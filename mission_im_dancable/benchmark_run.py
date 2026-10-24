"""CLI entry for --benchmark-pose: run each backend in its own worker
process (main venv for YOLO, .bench-venv for MediaPipe), one at a time.

Each worker decodes --source itself one frame at a time — nothing is
staged in shared memory or on disk. Results merge into
output/pose_benchmark/pose_benchmark.json keyed by backend name, so
separate one-backend invocations accumulate.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from . import benchmark
from .config import AppConfig, resolve_device

BENCH_VENV = Path(__file__).resolve().parent.parent / ".bench-venv"


def interpreter_for(backend: str) -> Path:
    """Pick the interpreter that has this backend's dependencies."""
    if backend.startswith("mediapipe"):
        mp_py = BENCH_VENV / "bin" / "python"
        return mp_py if mp_py.is_file() else Path(sys.executable)
    return Path(sys.executable)


def run(cfg: AppConfig) -> int:
    source = cfg.source
    assert source is not None
    meta = benchmark.read_video_meta(source)
    print(f"Video: {source} — {meta['width']}x{meta['height']} @ {meta['fps']:.1f}fps, "
          f"{meta['frame_count']} frames", flush=True)
    limit = cfg.max_frames or None
    print(f"Frames per backend: {limit or meta['frame_count']} "
          f"({'smoke' if limit and limit < meta['frame_count'] else 'full'} run)",
          flush=True)

    device = "cpu"
    try:
        import torch
        device = resolve_device(cfg.device, torch.cuda.is_available())
    except ImportError:
        pass

    out_dir = benchmark.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parent.parent
    results: list[dict] = []

    # Strictly sequential: one backend process lives at a time.
    for backend in [b.strip() for b in cfg.pose_backends.split(",") if b.strip()]:
        py = interpreter_for(backend)
        out_json = out_dir / f"bench_{backend}.json"
        cmd = [
            str(py), "-m", "mission_im_dancable._bench_worker",
            "--backend", backend,
            "--source", str(source),
            "--out", str(out_json),
            "--conf", str(cfg.confidence),
            "--device", device,
            "--fps", str(meta["fps"]),
            "--limit", str(cfg.max_frames),
            "--outdir", str(out_dir),
        ]
        if cfg.visualize_pose:
            cmd.append("--visualize")
        print(f"\n=== {backend} ({py}) ===", flush=True)
        proc = subprocess.run(cmd, cwd=repo, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"backend {backend} failed:\n{proc.stderr[-800:]}", flush=True)
            results.append({"backend": {"backend": backend},
                            "error": proc.stderr[-400:]})
            continue
        results.append(json.loads(out_json.read_text()))

    print("\n" + benchmark.summarize(results), flush=True)
    path = benchmark.save_results(results, meta)
    print(f"\nResults: {path}", flush=True)
    gpu = benchmark.nvidia_gpu_stats()
    if gpu:
        print(f"GPU: {gpu['gpu']} util={gpu['utilization']} mem={gpu['memory_used']}")
    return 0
