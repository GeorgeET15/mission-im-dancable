# Change log

Record each meaningful project change during the event, newest first. Update this file in the same commit as the change. Use a real timestamp with timezone, for example `YYYY-MM-DDTHH:MM:SS+05:30`.

## Event build log

## 2026-10-24T22:15:00+0530 — Milestone 2: reference choreography + practice mode

- Changed: Added `reference.py` (ReferenceChoreography model, offline pose extraction at configurable `--sample-fps`, `.npz` cache with staleness checks on video size/mtime, sample rate, model, confidence, and cache version), `moves.py` (JSON move segmentation with validation), `scoring.py` (confidence-aware normalized pose scoring, window matching, worst-mismatch + direction hints, mirroring, score smoothing, correction hysteresis), `practice.py` (side-by-side practice loop with per-move looping and N/P/Q controls). New CLI flags: `--source`, `--moves`, `--sample-fps`, `--window`, `--mirror`, `--cache-dir`. `.gitignore` now covers `cache/`; 34 new tests. Fixed two bugs caught by tests: cache compared the stored sample rate against itself instead of the requested one, and the correction tracker delayed the first correction by `hold_frames`.
- Why: Second milestone — turn the verified tracker into a usable practice demo: load a dance video, learn it move by move, get scored and corrected.
- Validation: `python -m pytest tests/` → 55 passed; extraction ran on a generated 6 s pan clip (60/60 samples detected a person, cached to `cache/`); second run hit the cache; practice mode ran ~12 s on the desktop display on CUDA without errors; `--help` renders all options.
- Limitations: Reference displays as a skeleton, not video frames; moves are manual only; extraction verified on a synthetic clip and webcam reads verified, but live practice against a real dancing person is still untested.

## 2026-10-24T22:05:00+0530 — Milestone 1: live pose tracker

- Changed: Added the `mission_im_dancable` package (`config`, `camera`, `pose`, `geometry`, `timing`, `render`, `app`, `__main__`) with a `python -m mission_im_dancable` entry point; 21 unit tests under `tests/`; `requirements.txt`, `requirements-torch-cuda.txt`, `requirements-torch-cpu.txt`; `.gitignore` rules for model weights and local media; populated the README project overview, run, and troubleshooting sections. Model weights live in `models/` (gitignored); Ultralytics `bus.jpg` moved to `tests/assets/`.
- Why: First milestone — verify the CUDA/pose-estimation stack and deliver the smallest complete feature before building game logic.
- Validation: `torch.cuda.is_available()` → True on RTX 3060 Laptop GPU; YOLO11n-Pose loaded and detected a person in `tests/assets/bus.jpg` at ~11 ms/frame on CUDA; `python -m pytest tests/` → 21 passed; webcam (`/dev/video0`) returned real 640x480 frames; full `python -m mission_im_dancable` loop ran for ~12 s on the desktop display without errors (cosmetic Qt font warnings only).
- Limitations: No reference choreography, scoring, or game UI yet. `pygame` has no Python 3.14 wheel, so `pygame-ce` 2.5.8 is used instead. Skeleton accuracy against a moving dancer still needs an on-person check at the event.

```text
## <actual timestamp> — <short change title>

- Changed: <feature or files changed>
- Why: <reason for the change>
- Validation: <commands or manual checks actually run, with results>
- Limitations: <known issues or "None identified in these checks">
```

## Template setup — 10 October 2026

- Changed: Added the event README, rules, agent guide, change log, AI usage record, submission checklist, Git ignore rules, and pull request checklist.
- Why: Give participants a consistent starting point for event rules, basic coding patterns, documented changes, and regular commits.
- Validation: Checked local Markdown links, event detail consistency, required documents, and diff whitespace before publishing.
- Limitations: Documentation scaffolding only; no participant project, application, or working demo is included. GitHub publication and template status are verified separately after creation.
