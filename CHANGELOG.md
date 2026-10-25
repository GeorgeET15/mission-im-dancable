# Change log

Record each meaningful project change during the event, newest first. Update this file in the same commit as the change. Use a real timestamp with timezone, for example `YYYY-MM-DDTHH:MM:SS+05:30`.

## Event build log

## 2026-10-25T09:20:00+05:30 — Play grace windows, auto-segmentation, stale-audio fix

- Changed: `PlaySession` regains the grace mechanic inside the continuous run — when a segment ends without passing, the run enters `GamePhase.GRACE`: the playhead loops inside that segment (reference + music loop too, via the drift re-anchor) for `grace_seconds` (5 s) while evals keep accruing to it. Passing early or timing out resumes at the boundary; each segment gets at most one grace; the last segment's grace runs before COMPLETE. `PlayScreen` shows a "LAST CHANCE!" popup on entry and a "COMPLETE THE MOVE — Ns" chip while gracing.
- Changed: `SelectScreen._start` auto-drafts segments for any video with no `.moves.json` (`suggest_boundaries` from the BeatGrid / pose-motion + `draft_moves`) so Learn's steps/N-P navigation and Play's per-move results get real steps instead of one "full choreography" blob. Drafts stay in-memory until saved in the editor; Learn header shows "auto-segments · E to refine". Real sidecar annotations are never overwritten.
- Fixed: stale soundtrack — `Music.load` was gated behind `music.loaded`, which stayed true for the previous video's WAV, so a newly analyzed video kept playing `dance_1`'s audio. `Music.load` is now idempotent-per-path and exposes `.path`; all three audio-sync sites (Play, Learn, Watch) load when `music.path != ctx.audio_path`.
- Validation: 230/230 tests (7 new: grace loop/timeout/early-recovery/once-only/no-grace-on-pass, auto-draft at start, annotations preserved).
- Limitations: grace pauses the *run* by looping the segment — the song effectively rewinds with it, matching the original move-by-move behavior.

## 2026-10-25T08:35:00+05:30 — UI click sound effects

- Changed: `sfx.py` — synthesized ~70 ms click blip (downward chirp, numpy → int16 stereo) played via `pygame.mixer.Sound` on whatever channel is free; `Sfx` wrapper is a safe no-op without a mixer and scales to the music-volume setting. `Button` gained a class-level `on_press` hook fired inside `press()` — `app.py` binds it to `Sfx.click`, so every button (mouse or keyboard-activated) clicks without widgets knowing about audio. Hook errors are swallowed so sfx can never break input.
- Why: request for button-click feedback matching the arcade feel.
- Validation: 223/223 tests (4 new: blip shape, disabled no-ops, hook fires, hook-error resilience).

## 2026-10-25T08:15:00+05:30 — Title readability + synthesized lobby music

- Changed: Title font switched from bold mono to a clean sans stack (`ubuntusans,dejavusans,liberationsans`) — the mono glyphs plus glow halo were hard to read. Title text is now "MISSION I'M-DANCABLE" (apostrophe kept, wording unchanged per owner). SelectScreen start button widened 190→240 px so "START LEARNING"/"START DANCING" no longer overflow their border.
- Changed: `lobby.py` — synthesized looping menu music (96 BPM Am–F–C–G pluck arpeggio, ~10 s loop, numpy → int16 stereo, loop-edge fades). `LobbyMusic` plays it on a dedicated mixer **Channel** — separate from `pygame.mixer.music`, which owns the dance soundtrack — so the two never collide. The app loop starts it on menu screens (Home/Select/Browse/Settings/Scores) and stops it on every gameplay screen; volume scales from the music-volume setting; fully no-op when the mixer is unavailable.
- Why: readability complaint on the title; request for lobby music without adding a bundled/licensed audio asset.
- Validation: 219/219 tests (3 new: stereo int16 shape/duration, loop-end fade vs peak, disabled no-ops); title rendered and reviewed headless.
- Limitations: track is a generated synth loop — no user-supplied lobby track support yet; audition of the actual sound not verified by ear in headless runs.

## 2026-10-25T07:40:00+05:30 — File browser redesign: preview pane + metadata

- Changed: `BrowseScreen` rewritten as two panes. Left: directory list with per-video duration + size on each row. Right: PREVIEW pane showing a thumbnail grabbed ~15% into the highlighted video (falls back to first frame, then "(no frame)"), duration/resolution/fps/size, and readiness badges — `SEGMENTS ✓` when `<stem>.moves.json` exists, `POSES CACHED ✓` when a matching pose cache file exists (fast ANALYZE). Video info and thumbnails are cached per path; thumbnail load is debounced (~0.2 s after the selection settles) so arrow-key scrolling stays responsive and `draw` never blocks on cv2.
- Changed: click behaviour is now select-then-open (first click highlights + previews, second opens); keyboard unchanged. Quick-jump buttons: UP / HOME / REPO (the directory the browser opened in) / CANCEL. Left arrow still goes to the parent dir.
- Why: the old bare name list gave no way to tell a usable dance clip from a random file — no preview, no duration, no indication which videos were already analyzed.
- Validation: 215/215 tests (3 new: probe never-raises + caching, click-selects-then-opens, debounced thumbnail cache); headless render of the repo dir with `dance_1.mp4` verified — real frame, correct metadata, both badges shown.
- Limitations: probes run synchronously (~5 ms/file on first paint of a directory); no multi-select or search.

## 2026-10-25T07:05:00+05:30 — In-game segment editor + auto-suggestions

- Changed: `editor.py` — `SegmentDraft` working copy of the move list (split at playhead, merge with next, ±0.1 s boundary nudge, reorder, rename, `from_draft` flag) and `suggest_boundaries`: cuts every 8th beat when the BeatGrid is usable, else falls back to valleys in cached pose-motion energy. `draft_moves` guarantees non-overlapping, ≥0.2 s segments.
- Changed: `moves.py` — `validate_segments` (empty/overlap/out-of-range/empty-name checks), `save_dance_file` (atomic write that preserves existing id/title/difficulty meta), `load_dance_meta` lenient reader.
- Changed: `ui/learn.py` — E key / EDIT SEGS button opens the editor inside Learn Mode only: B split · X merge · [ ] nudge · ↑/↓ select · PgUp/Dn reorder · N name (TextField) · ENTER preview-loops the segment · G auto-draft (confirm-on-repeat when edits exist) · A accept suggestions (gold-tinted until accepted) · S save (confirm-on-repeat when overwriting an existing file) · E/ESC exits (confirm-on-repeat to discard). Saving reloads `ctx.moves`/`ctx.dance`/session moves so navigation, looping, and Play results share one segment list. Videos without a sidecar still work (default full-video segment) and editing one creates `<stem>.moves.json` on save.
- Why: every imported video needs a way to get real, verified segments without hand-editing JSON.
- Validation: 17 new tests in `test_editor.py` (split/merge/nudge/reorder/rename/validation/save-reload/meta-preservation/invalid sidecar/beat+motion suggestions/edit-mode flow/overwrite confirm) → 212/212 pass.
- Limitations: editor is keyboard-driven (no drag); suggestions are phrase-level drafts, not verified moves — the UI labels them SUGGESTED until accepted.

## 2026-10-25T06:10:00+05:30 — Learn Mode + Play Mode two-mode architecture

- Changed: Home screen now offers **LEARN** and **PLAY** as the two primary modes sharing one dance library. `ctx.mode` is `"learn"`/`"play"`; the select screen's mode toggle is gone (mode comes from Home), its start button routes to `LearnScreen` or `ReadyScreen`→`PlayScreen`, and its header/info show mode + difficulty.
- Changed: `PlaySession` (game.py) — continuous scored run from the first segment's start to the last's end; same fixed-interval scoring (0.25 s), same 100×multiplier points, combo and pass rules; per-segment eval bucketing via `moves.move_index_at`; `restart()` for full-run retries; run ends immediately at `end_time` rather than on an eval boundary. `GameSession` (move-by-move briefings/grace) retained for tests but no longer used by the UI; `PlayScreen` is now Play-Mode-only and its soundtrack plays start-to-finish with drift re-anchoring.
- Changed: `learn.py` — pygame-free `LearnSession`: play/pause, exact seek/skip, frame stepping, 0.25–1× speeds, segment select/prev/next, loop off→segment→all, `segment_state` (done/selected/active/upcoming). `ui/learn.py` — `LearnScreen`: reference vs live skeleton panels, background camera open that degrades to watch-only on failure (R retries), non-competitive match % + correction hint, seekable `TimelineBar` with per-segment blocks + progress dots + playhead + time readout, button row (play/speed/seg±/loop/mark/mirror/back), auto-"learning" marks on first visit, K cycles learning→practicing→mastered→clear. Audio follows the playhead only at 1× speed (pygame can't time-stretch).
- Changed: `widgets.TimelineBar` reusable seekable timeline; `dance.py` `Dance` model (id/title/video/duration/fps/moves/difficulty); moves files now accept optional `id`/`title`/`difficulty` + per-move `instruction` (`load_dance_file` returns moves+meta, `load_moves` unchanged); `persist.py` stores per-dance move statuses in `progress.json` under the user data dir; `eval_against_reference` shared by both screens so the pose comparison is identical.
- Changed: `dance_1.moves.json` rewritten as a real dance file — 6 segments with names + instructions grounded in frame-by-frame inspection of `dance_1.mp4` (contact sheets at 0.4 s intervals).
- Why: Spec asked for a Learn/Play split — free-form study vs one continuous scored run — sharing the dance library, reference pipeline, and pose engine without duplicating the game.
- Validation: `pytest tests/` → 195 passed (new: `test_playsession` 13, `test_learn` 16, `test_dance` 8, `test_progress` 6, `test_learn_flow` 14 headless screen tests; `test_flow`/`test_gameplay` updated for PlaySession). Headless end-to-end smoke drove Home→LEARN→analyze `dance_1.mp4`→LearnScreen (play/seek/segment/loop/speed/marks verified, screenshot reviewed) and Home→PLAY→Ready→PlayScreen→Results flow; camera-open and camera-failure paths both exercised (webcam busy → watch-only + CAMERA PROBLEM overlay, no crash).
- Limitations: no audio-rate scaling below 1× speed (muted instead); segment boundaries remain manual annotations; real-dancer playthrough still untested; PlaySession's mid-run move results are informational only (no retry affordance, by design).

## 2026-10-25T04:40:00+05:30 — UI/UX overhaul: readiness flow, audio, browse, HUD polish

- Changed: Large `ui/` overhaul. New `ReadyScreen` between select and gameplay — camera opens on a worker thread, live preview + skeleton, "step into frame" detection hold, "got you" beat, 3-2-1 countdown that pauses if the pose drops, error state with RETRY/BACK. New `BrowseScreen` in-game file browser (dirs + video files, keyboard/mouse/wheel, scroll) wired to BROWSE. New `audio.py` — ffmpeg decodes the video's audio to a cached WAV; `Music` (pygame.mixer.music) plays each move's `[start,end)` segment in performance, loops per move in practice, full track in WATCH, with pause/unpause, seek re-anchoring on drift >0.25 s, and graceful silence when no audio/device. `GameSession` unchanged.
- Changed: Navigation fixes — `Navigator.swap()` (pop+push without re-entering the suspended screen) drives Ready→Play/Watch→Ready/Results→retry handoffs so the Home/Select stack survives; Play `Q` pops when stacked, else lands on Select; Select back pops to Home or replaces at depth 1 (no more empty-stack black screen); Home gains ESC→quit confirm; app loop exits on stack underflow.
- Changed: Shared webcam on `ctx.cap` (opened once by ReadyScreen, reused by PlayScreen/retries, released on menu entry and at teardown). Camera failure in play shows a CAMERA PROBLEM overlay; `R` re-runs readiness.
- Changed: HUD — REFERENCE/YOU panel labels, accuracy bar with 60% pass tick, move progress + elapsed/duration, whole-routine footer progress bar, pose-loss chip ("POSE LOST — STEP INTO FRAME") replacing MISS-popup spam (untracked evals no longer flash MISS — scoring unchanged), camera-error overlay. Footer text sits on dim strips above the equalizer.
- Changed: Select screen — detected-video cards show duration + best score, click selects and auto-analyzes, selected state highlighted, first-pose preview, empty-dir guidance. Results — MISSION COMPLETE header, dominant score, NEW PERSONAL BEST badge vs persisted per-video best, accuracy/combo/moves stat cards, per-move rows, R retry. Settings — mirror + fullscreen toggles, camera-index/match-window/volume steppers, saved on every change (new persisted keys `fullscreen`, `window_s`, `volume`); F11 toggles fullscreen anywhere; resizable letterboxing retained. Fonts re-hierarchied (title 64 > big 56 > med 24 > sm 18). Skeleton glow layer now reuses one cached SRCALPHA surface.
- Why: Turn the functional prototype into a coherent game flow a new player can navigate without instructions; play the reference audio during gameplay; browse arbitrary directories for videos.
- Validation: `pytest tests/` → 142 passed (new `test_flow.py`: nav swap, ready detect/countdown/abort/cancel, auto-start, no-MISS-on-pose-loss, camera-R-retry, back-at-depth-1, results personal-best, draw-before-update regression, browse navigate/pick/cancel, music sync incl. disabled no-ops). `--game` smoke-run on the live display: Home→Select→analyze→Watch→Ready→(camera busy → error state) exercised without crashes. Headless PNG screenshots of all 9 screens reviewed.
- Limitations: audio/visual sync unverified by ear (verified by position bookkeeping only); scoring against a real dancer still untested; zenity exists on this machine but the in-game browser is used unconditionally for consistency.

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
