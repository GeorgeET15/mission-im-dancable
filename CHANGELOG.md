# Change log

Record each meaningful project change during the event, newest first. Update this file in the same commit as the change. Use a real timestamp with timezone, for example `YYYY-MM-DDTHH:MM:SS+05:30`.

## Event build log

## 2026-10-24T22:30:00+0530 — Animated reference skeleton preview

- Changed: Added `pose_at()` to reference.py — timestamp-based interpolation between adjacent pose samples with per-landmark missing-data handling (never invents landmarks). Added `playback.py` with `PlaybackClock` (elapsed wall-time playback: play/pause/seek/skip/restart, auto-stop at end, speed independent of tick rate). Added `preview.py` — OpenCV playback window showing the animated reference beside the live camera skeleton; SPACE pause, ←/→ seek ±2 s, R restart, ENTER hands the SAME camera into `run_practice_loop`. Refactored `run_practice` into `prepare()` + `run_practice_loop()` so preview and practice share resources (no second camera open). New flags `--preview` and `--no-cam` with cross-flag validation. 16 new tests.
- Why: Let users watch the choreography's skeleton at original speed before performing it — the first acceptance criterion for the video-to-practice flow.
- Validation: `python -m pytest tests/` → 95 passed (clock timing/pause/seek/end, interpolation incl. missing landmarks, panel rendering, mirror). `--preview` ran ~12 s on the display on CUDA (with and without `--no-cam`), cache-hit load, no errors.
- Limitations: Arrow-key codes for seek are backend-dependent (81/83 and 2/3 handled); interactive key flow unverified headlessly; original video pixels still not shown (skeleton only) — documented trade-off.

## 2026-10-24T22:25:00+0530 — Milestone 3: rhythm-game layer + pygame UI

- Changed: Added `game.py` (UI-independent GameSession: BRIEFING/ACTIVE/RESULT/COMPLETE phases, interval-gated scoring at 0.25 s so points can't scale with FPS, duplicate-frame protection via frame_id, combo/multiplier, pass/fail with mean-accuracy + reliable-fraction thresholds, retry and mission scorecard). Extended `scoring.py` with joint-angle comparison (`JOINTS`, `angle_errors`, `PoseAnalysis`, `analyze_pose`, `describe_angle_correction`, `joint_angle_deg`) and widened CorrectionTracker to string hints. Added `game_ui.py` — pygame-ce mission-control UI (reference vs live panels, score/combo/accuracy HUD, briefing/result/pause/scorecard screens, loading + fatal-error states). Shared helpers: `window_indices`/`nearest_index` in reference.py, `fit_keypoints` in render.py (practice.py refactored to reuse). New `--game` flag; OpenCV tracker and practice remain as fallbacks. 24 new tests.
- Why: Third milestone — turn the practice prototype into a playable rhythm game with score, combo, and mission progression.
- Validation: `python -m pytest tests/` → 79 passed (interval gating, FPS-independence, duplicate-frame rejection, combo math, pass/retry, scorecard totals, angle errors incl. mirroring and missing landmarks). `--game` ran ~15 s on the desktop display on CUDA without errors. Interactive key flow (SPACE/R/P/M) untested beyond code review.
- Limitations: No full playthrough against a real dancer yet; reference still renders as skeleton only; move segmentation still manual; pause/overlay screens verified only by smoke run.

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
