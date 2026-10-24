# AI usage record

Use Codex during the hackathon and briefly document its actual contribution here. Add a row after each meaningful task or milestone. Keep summaries concise and omit secrets, private data, and full chat transcripts.

## Event build log

Mark human review as pending until someone has reviewed the contribution.

| Date and time with timezone | Tool / model, if known | Task | Assistance and affected files | Human review / corrections | Validation and result |
| --- | --- | --- | --- | --- | --- |
| 2026-10-24T23:40:00+0530 | Devin (SWE-2) | Low-poly geometric dancer | Authored `ui/dancer.py` (procedural faceted avatar, facet shading, painter sort, lerp smoothing), `--avatar` flag, 8 tests. | Pending | 124/124 tests pass; arcade ran on display; avatar look on real footage untested. |
| 2026-10-24T23:20:00+0530 | Devin (SWE-2) | Beat synchronization | Authored `beats.py` (ffmpeg decode, onset detection, cache, BeatGrid), UI wiring for beat-pulsed visuals, 7 tests. | Pending | 116/116 tests pass; 38 beats detected on dance_1.mp4; audio-visual sync quality untested. |
| 2026-10-24T23:05:00+0530 | Devin (SWE-2) | Real-time gameplay tests + crash fixes | Authored headless gameplay tests (fake camera/tracker/clock) covering cadence, unreliable poses, camera failure, pause, completion handoff; fixed TextField.contains and numpy-coord draw crashes found by user. | Pending | 109/109 tests pass; user-verified select screen works post-fix. |
| 2026-10-24T22:50:00+0530 | Devin (SWE-2) | Dance-arcade pygame UI | Authored the `ui/` package (theme, widgets, navigator, six screens, app loop), `persist.py` settings/high-scores, judgement/grade logic, extraction progress callback; replaced `game_ui.py`; 8 new tests + docs. | Pending | 103/103 tests pass; UI launched on display ~15 s without errors; interactive navigation unverified. |
| 2026-10-24T22:30:00+0530 | Devin (SWE-2) | Animated reference skeleton preview | Authored `pose_at` interpolation, `PlaybackClock`, `preview.py` playback window with camera handoff into practice (`prepare`/`run_practice_loop` refactor), `--preview`/`--no-cam` flags, 16 tests, docs updates. | Pending | 95/95 tests pass; preview ran on display with and without camera; key controls unverified headlessly. |
| 2026-10-24T22:25:00+0530 | Devin (SWE-2) | Milestone 3: game state + pygame UI | Authored `game.py` (session state machine, interval scoring, combo, scorecard), joint-angle analysis in `scoring.py`, `game_ui.py` (pygame-ce screens + HUD), `--game` flag, 24 new tests, docs updates. Human-verified behavior: none yet — interactive play untested. | Pending | 79/79 tests pass; game window ran ~15 s on display with no errors; FPS-independence and dedup verified by simulation tests. |
| 2026-10-24T22:15:00+0530 | Devin (SWE-2) | Milestone 2: reference extraction + practice mode | Authored `reference.py`, `moves.py`, `scoring.py`, `practice.py`, config flags, `.gitignore` cache rule, 34 new tests, and README/CHANGELOG updates; generated a synthetic pan clip for end-to-end verification. Tests caught and Devin fixed a cache staleness bug (stored-vs-requested sample rate) and a first-correction delay bug. | Pending | 55/55 tests pass; extraction + cache-hit verified on synthetic clip (60/60 detections); practice window ran on display with no errors; real-dancer accuracy untested. |
| 2026-10-24T22:05:00+0530 | Devin (SWE-2) | Environment setup + milestone 1 live pose tracker | Authored the `mission_im_dancable` package, tests, requirements files, `.gitignore` additions, and README updates; resolved Python 3.14 compatibility (kept 3.14 venv, used `pygame-ce` because `pygame` lacks cp314 wheels); copied installed packages from the prior `~/dance-teacher` venv at user direction. | Pending | CUDA verified on RTX 3060; model loads and runs inference (~11 ms) on test image; 21/21 tests pass; live camera loop smoke-tested end-to-end on the desktop display. |

## Template preparation

On 10 October 2026, Codex prepared this documentation template from the organizer's supplied rules and checked the linked Luma page for the event name and dates. This covered `README.md`, `RULES.md`, `AGENTS.md`, `CHANGELOG.md`, `AI_USAGE.md`, `SUBMISSION.md`, `.gitignore`, and the pull request checklist. Agent validation covered local links, event consistency, required documents, and diff whitespace; organizer review is pending.

This preparation is separate from participant project implementation and participant AI usage during the event.
