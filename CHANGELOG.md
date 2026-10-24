# Change log

Record each meaningful project change during the event, newest first. Update this file in the same commit as the change. Use a real timestamp with timezone, for example `YYYY-MM-DDTHH:MM:SS+05:30`.

## Event build log

## 2026-10-25T04:05:00+05:30 — Grace window loops the move instead of freezing

- Changed: During `GamePhase.GRACE` `session.time` now wraps within `[move.start, move.end]` — the reference skeleton replays the move cyclically while the player finishes, rather than holding a static final pose. `session.time` jumps to `move.start` on entry; scoring clock and countdown unchanged.
- Why: User decision — a dance move is a motion, so the target should keep moving; matching a single frozen pose would be trivial and unrepresentative.
- Validation: `pytest tests/` → 122 passed; grace tests now assert time stays in range and wraps (looping, not frozen).
- Limitations: none beyond existing grace scope (performance mode only).

## 2026-10-25T03:55:00+05:30 — Freeze reference during the completion window

- Changed: During `GamePhase.GRACE` `session.time` now clamps at `move.end` — the reference skeleton holds the move's final pose instead of playing ahead into the next move (typing-game behavior). Scoring intervals run on a separate `_eval_elapsed` clock so evaluations still commit during the window; `_grace_elapsed` tracks the countdown.
- Why: User feedback — the dancer kept stepping into the next move during grace, which made the window meaningless.
- Validation: `pytest tests/` → 122 passed; new assertions verify `session.time == move.end` throughout the window.
- Limitations: none beyond the existing grace scope (performance mode only).

## 2026-10-25T03:40:00+05:30 — Full-dance WATCH mode + 5s move-completion window

- Changed: New `WatchScreen` (pygame) plays the whole choreography skeleton end-to-end via `PlaybackClock` — SPACE pause, arrows seek, R restart, M mirror, ENTER starts dancing; `WATCH` button on the select screen enabled after analyze. `GamePhase.GRACE` + `GameConfig.grace_seconds` (default 5 s): when a move ends while failing, the session enters a countdown window instead of instant MOVE FAILED — recovery during the window passes early; `grace_seconds=0` restores old behavior. Play screen shows a "COMPLETE THE MOVE — Xs" banner during the window.
- Why: User request — view the full dance before playing; soften abrupt per-move failure.
- Validation: `pytest tests/` → 122 passed (6 new grace tests incl. early pass, expiry, pause, retry, grace=0; 3 older fail-path tests pinned to grace_seconds=0). UI wiring reviewed; not yet exercised on the live display.
- Limitations: GRACE prompts only in performance mode (PracticeSession unchanged); watch screen shows the reference only, no camera.

## 2026-10-25T03:20:00+05:30 — Remove the 3D dancer (skeleton-only gameplay)

- Changed: Deleted `avatar.py`, `avatar_render.py`, `ui/sprite.py`, `tests/test_avatar.py`, `tests/test_sprite.py`. Removed from `ui/screens.py`: `render_choreography`/`load_frames` imports, `ctx.amy`/`ctx.avatar_dir`, the analyze worker's Blender render call, the Amy-vs-skeleton branch in the play screen (skeleton-only now), and the Settings "AMY sprite reference" toggle. `persist.py` default settings no longer include `amy` (old saved keys are harmlessly ignored). `assets/blender/` contents (`.blend`, `.blend1`, rendered frames, preview/check PNGs) moved to `~/mission-im-dancable-model-archive/blender/`; `cache/amy/` deleted.
- Why: User decision — return to the original skeleton-only reference visualization; the Blender/Mixamo path was an experiment.
- Validation: `pytest tests/` → 116 passed (avatar/sprite tests removed with the feature). No remaining code references to amy/sprite/blender/avatar/mixamo. Skeleton reference panel (`pose_at` + `_neon_skeleton`) is the only reference renderer.
- Limitations: Saved settings files may still contain an `amy` key — ignored by `load_settings`. Blender no longer needed for any feature.

## 2026-10-25T03:00:00+05:30 — Single-model cleanup: keep only YOLO11l-Pose

- Changed: Default and only gameplay model is now `yolo11l-pose.pt` (`DEFAULT_MODEL_PATH`, `--model` still accepted for explicit paths). Removed all benchmark infrastructure: `mediapipe_pose.py`, `benchmark.py`, `benchmark_run.py`, `_bench_worker.py`, `tests/test_benchmark.py`, the `--benchmark-pose`/`--visualize-pose`/`--pose-backends`/`--max-frames` flags, `PoseTracker.close()`, benchmark gitignore entries, the `.bench-venv`, and `output/pose_benchmark/` artifacts. Other weights moved outside the repo to `~/mission-im-dancable-model-archive/` (yolo11n/11m/11x/26m/26l + pose_landmarker_heavy.task, all SHA-256 verified before removal). `PoseTracker.predict` keeps `inference_mode` + CUDA sync timing.
- Why: Simplify to one model, one gameplay path — benchmarks proved yolo11l is the best measured option (100% detection on dance_1.mp4, ~72 FPS on RTX 3060) so it becomes the default.
- Validation: `pytest tests/` → 133 passed. `sha256sum` verified retained 11l (61921abe…) and all 6 archived files before deletion. `git ls-files models/` empty — no weights tracked.
- Limitations: Cache fingerprints include the model name, so first-run re-extraction happens once per video; `--model` can still point at other weights by explicit user choice.

## 2026-10-25T02:30:00+05:30 — YOLO pose variant benchmark (11m/11l/11x/26m/26l)

- Changed: `benchmark.py` gained `yolo-<variant>` backend resolution (`models/yolo<variant>-pose.pt`, explicit `--model` still wins), per-model SHA-256 + post-warmup CUDA-device verification (predictor's inner module), missing-frame index recording, inference-phase GPU-peak reset; `pose.py` wraps predict in `torch.cuda.synchronize()` for honest GPU timing; `benchmark_run.py` forwards `--model` only when explicitly set. Downloaded official v8.4.0 weights for yolo11m/l/x-pose and yolo26m/l-pose (gitignored). 4 new tests.
- Why: Check whether a larger or newer pose model improves dance tracking while staying real-time on the RTX 3060 — baseline 11n misses 5 frames during the backlit/hazy intro.
- Validation: `pytest tests/` → 157 passed. Full 354-frame CUDA run per model, one at a time (`device_verified: true`, honest sync timing): 11n 8.5 ms / 98.6% / gap 5; 11m 11.0 ms / 99.1% / gap 1; **11l 13.9 ms / 100% / gap 0**; 11x 25.3 ms / 100% / gap 0; 26m 11.6 ms / 98.3% / gap 6; 26l 14.0 ms / 98.6% / gap 5. All misses are frames 5–10 (spotlight + stage haze); overlay check shows 11l recovers a clean skeleton there.
- Limitations: Single clip; misses are an intro edge case, so 100% detection reflects this video only; jitter includes real motion; recommendation (11l) is opt-in — code default unchanged; no scoring-threshold recalibration tested; YOLO26 offers no benefit on this clip.

## 2026-10-25T02:00:00+05:30 — MediaPipe vs YOLO pose backend benchmark

- Changed: Added `mediapipe_pose.py` (VIDEO-mode `PoseLandmarker` adapter; MediaPipe 33→COCO-17 mapping with out-of-frame/NaN rejection, world-landmark capture, pluggable cpu/gpu delegate), `benchmark.py` (streaming `iter_frames`, per-frame latency/reliability/jitter/world stats, peak RSS + CUDA peak, merge-on-save JSON), `benchmark_run.py` + `_bench_worker.py` (one backend per worker process — `.venv` for YOLO, `.bench-venv` for MediaPipe — strictly sequential). New CLI: `--benchmark-pose`, `--pose-backends`, `--visualize-pose`, `--max-frames` (default 30 smoke, 0 = full). `pose.py` gained `torch.inference_mode()` + `close()`. `.gitignore` covers `.bench-venv/` and `*.task`. 19 new tests.
- Why: Evaluate MediaPipe Pose Landmarker Heavy against YOLO11n-Pose for latency/reliability/3D-animation suitability without touching the default backend. Rewrote the first attempt's staging (whole clip → list + npz → per-worker decompress) after it OOM-killed a 14 GB machine; the streaming design holds ~1 frame at a time.
- Validation: `pytest tests/` → 153 passed (mapping, bounds, timestamps, cleanup, merge, streaming, bounded viz via synthetic clips). Full 354-frame run per backend, one at a time: yolo/CUDA 8.9 ms mean, 98.6% detected, peak RSS 1.68 GB; mediapipe-heavy CPU 46.9 ms, 94.8%; mediapipe-heavy GL delegate 21.8 ms, 94.8% — EGL binds the **AMD iGPU**, not the RTX 3060. Overlay videos written for all three; world landmarks saved to npz (fingers/heels least stable, hips trivially stable at the origin).
- Limitations: MediaPipe GPU delegate ≠ NVIDIA; world coords are model estimates; jitter includes real motion; LIVE_STREAM mode not implemented (VIDEO only); benchmark on one clip only.

## 2026-10-25T00:58:53+05:30 — Drive Amy from the reference skeleton

- Changed: `avatar.py` maps confidence-filtered YOLO17 reference samples to planar directions for Amy's torso, neck, arms, and legs, plus limited hip translation; `avatar_render.py` applies those directions to the skinned Mixamo rig in background Blender and writes one transparent PNG per sampled timestamp. Renders are cached under `cache/amy/` using pose, rig, and confidence fingerprints; incomplete renders are ignored. `ui/screens.py` generates video-specific frames in the selection worker and falls back to the skeleton on failure; `ui/sprite.py` selects frames by reference timestamps (pause, retries, and mirror included), not the old canned 48-frame loop. Persisted Amy preference now loads correctly in `persist.py`. Added `tests/test_avatar.py` and a preference regression test.
- Why: The user reported Amy was T-posed/unrelated to the dance video. Video-specific bone targets replace the decorative loop as the game's reference visualization without changing pose scoring or camera inference.
- Validation: `compileall` and `pytest tests/` passed (133 tests). Headless Blender rendered all 118 samples from `dance_1.mp4` (320x400 RGBA), with no empty/cropped frames; frames 11 and 41 have different poses. Pygame panel drew cached frame 11 at 1 s and frame 61 at 6 s. `--game` opened on the desktop display for 8 s without errors (no interactive playthrough).
- Limitations: Planar single-camera approximation, not full 3D motion capture: occlusion, depth, spins, contact, and fingers are not recovered. Full rendering happens once per video/settings before play and can take time. Scoring against a real dancer is still unverified; video and generated assets remain gitignored.

## 2026-10-25T00:20:00+0530 — Amy sprite reference (milestone 1)

- Changed: Added `ui/sprite.py` — `SpriteAnim`/`load_frames`/`frame_index_at` play pre-rendered Blender frames, frame index derived from the shared session clock (pause/seek/restart free). PlayScreen draws Amy in the reference panel when `settings["amy"]` (default on, Settings toggle) and frames exist; neon skeleton is the fallback. Rendered 48 transparent frames (320×400 RGBA, EEVEE, platform hidden) from `mission_dancer.blend` via headless `blender -b`. Removed the earlier geometric dancer attempt (`ui/dancer.py`, `--avatar`, tests) — superseded by real Amy. `assets/` added to .gitignore (28MB blend + frames stay local). 6 new tests.
- Why: Prove Amy can be rendered headless and displayed in-game before attempting pose retargeting.
- Validation: `pytest` → 122 passed (index math, looping, pause/seek determinism, missing-dir fallback, load roundtrip). Headless draw verified: 48 frames load, 18.5k px blitted, index 12 at t=0.5. `--game` ran ~15 s on display, no errors.
- Limitations: Amy plays her baked 48-frame sway loop — she does NOT reproduce reference choreography yet (retargeting is a later milestone). Sprite seen only via headless draw test, not live play screen.

## 2026-10-24T23:20:00+0530 — Beat synchronization

- Changed: Added `beats.py` — ffmpeg-based PCM decode, onset-energy beat detection (60–180 BPM autocorrelation), npz cache with staleness checks, and `BeatGrid` (pulse/phase/nearest queries). Wired into the select-screen analyze thread (no render-loop cost) and PlayScreen (beat-pulsed reference panel border, judgement popup kick). Scoring untouched — visuals only. Fixed `pulse()` off-by-one for beats exactly at t. 7 new tests.
- Why: Make the arcade feel like a rhythm game — visuals breathe with the music.
- Validation: `pytest` → 116 passed (synthetic-beat grid queries, cache staleness, silence/no-audio, 120 BPM synthetic clicks). Real extraction on `dance_1.mp4` → 38 beats, ~0.31 s mean spacing. `--game` smoke pending re-check.
- Limitations: Heuristic onset detection — not librosa-grade; may double-count on busy tracks. Beat-accuracy correlation with actual music untested. No audio device verification done.

## 2026-10-24T23:05:00+0530 — Real-time gameplay verification + fixes

- Changed: Added `tests/test_gameplay.py` — 6 headless tests driving `PlayScreen.update` with fake camera/tracker/clock: practice mode dt-advance + pause, performance scoring cadence, unreliable-pose combo breaks, camera-failure-as-miss, pause committing nothing, and the RESULT→COMPLETE→ResultsScreen handoff. Fixed `TextField.contains` crash and numpy-float pygame draw crash on the select screen.
- Why: Prove the continuous scoring path end-to-end without a webcam, per milestone requirement that score/combo/judgements reflect actual evaluated poses.
- Validation: `pytest` → 109 passed; `--game` ran ~12 s on display; clicks on the select screen verified working by user (after contains fix).
- Limitations: Live webcam gameplay with a real dancer still untested; results flow verified only via headless test.

## 2026-10-24T22:50:00+0530 — Dance-arcade pygame UI

- Changed: Replaced the mission-control `game_ui.py` with a `ui/` package — `theme.py` (neon palette, gradient, dance floor, spotlights, equalizer, particles, glow text), `widgets.py` (Button/Toggle/ProgressBar/TextField/FocusGroup), `nav.py` (pygame-free Navigator with fade + input lock), `screens.py` (Home/Select/Play/Results/Settings/Scores + AppContext), `app.py` (`run_arcade` with resizable letterboxed canvas). Added `persist.py` (settings + high scores under XDG config/data dirs), `Judgement`/`judge()`/`grade_for()` in game.py, `on_progress` callback for extraction (background-threaded on the select screen). `--game` no longer requires `--source` — the video is picked in the UI. 8 new tests.
- Why: Turn the working prototype into a real rhythm game: home → dance select → play → results with score, combo, judgements, and persistence.
- Validation: `python -m pytest tests/` → 103 passed (navigator transitions + input lock, settings/scores persistence, judgement/grade mapping). `--game --source dance_1.mp4` ran ~15 s on the display with no errors; menu navigation and clicks unverified beyond code review.
- Limitations: Interactive menu/playthrough untested (no display input automation); tkinter absent so browse uses the path-field fallback; practice mode inside the arcade UI mirrors the GameSession layout without scoring.

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
