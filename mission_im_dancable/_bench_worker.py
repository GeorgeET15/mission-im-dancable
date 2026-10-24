"""Benchmark worker: runs ONE backend over a video, streaming decode.

Invoked as `python -m mission_im_dancable._bench_worker --backend NAME
--source VIDEO --out r.json [--limit N] [--visualize]`. Run with whichever
interpreter has that backend's dependencies (main venv for yolo,
.bench-venv for mediapipe). One backend per process keeps memory bounded:
the model and its interpreter die with the process.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--model", type=Path, default=None)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--limit", type=int, default=30,
                   help="max frames decoded including warmup; 0 = whole video")
    p.add_argument("--outdir", type=Path, required=True,
                   help="persistent output dir for overlays/world npz")
    p.add_argument("--visualize", action="store_true")
    args = p.parse_args()

    from . import benchmark

    args.outdir.mkdir(parents=True, exist_ok=True)
    viz = args.outdir / f"visual_{args.backend}.mp4" if args.visualize else None
    result = benchmark.run_benchmark(
        args.source,
        args.backend,
        args.conf,
        device=args.device,
        model_path=args.model,
        fps=args.fps,
        warmup=args.warmup,
        limit=args.limit or None,
        visualize_path=viz,
    )
    world_seq = result.pop("_world_seq", None)
    if world_seq is not None:
        np.savez_compressed(args.outdir / f"world_{args.backend}.npz",
                            world=world_seq)
        del world_seq
    args.out.write_text(json.dumps(result))

    # Let the interpreter exit cleanly; one cache clear at the end is fine
    # (never per-frame). Releases CUDA blocks before process teardown.
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    print(f"{args.backend}: wrote {args.out}")


if __name__ == "__main__":
    main()
