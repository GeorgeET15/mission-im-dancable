# Mission Im-Dancable

*Your mission, should you choose to accept it.*

An AI-powered dance-teaching rhythm game built at DevDay Exchange Community Hackathon: Kochi (10–11 October 2026). Learn choreography one move at a time, match your skeleton to reference poses, and score combos like a rhythm game.

**[Event and registration on Luma](https://luma.com/z10nz8gp)** · **[Event rules](RULES.md)** · **[Agent instructions](AGENTS.md)**

## Event details

| Detail | Information |
| --- | --- |
| Venue | JAIN University, Kochi |
| Location | [Open the venue map](https://maps.app.goo.gl/Eme7gR7NDXSNRANu5) |
| Rooms | 2008 and 2009 |
| Check-in | 10 October 2026, 6:00 PM IST |
| Kick-off | 10 October 2026, 6:30 PM IST |
| Wrap-up | 11 October 2026, by 10:00 AM IST |
| Participation | Solo or a team of up to 4 participants |
| Theme | Open — build anything you like |

Keep your approved Luma QR code ready for entry, and bring your laptop and charger. Eligible checked-in participants will receive Codex credits and OpenAI API credits.

## Project overview

- **Project name:** Mission Im-Dancable
- **Short description:** A rhythm-game-style dance trainer. It tracks your body with pose estimation, teaches choreography one move at a time, and scores how closely you match the reference — turning "learn to dance" into a mission to beat.
- **Team:** GeorgeET15
- **Working demo:** Local demo — see run instructions below.
- **Technology stack:** Python 3.14, PyTorch (CUDA cu128), Ultralytics YOLO11l-Pose, OpenCV, NumPy, pygame-ce, pytest.

### Run the project

1. **Prerequisites:** Python 3.14, a webcam, and (for GPU inference) an NVIDIA GPU with a CUDA-12.8-capable driver. CPU fallback works too — just slower.
2. **Create the environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. **Install dependencies** — CUDA torch first so ultralytics doesn't pull a CPU-only build:
   ```bash
   pip install -r requirements-torch-cuda.txt   # NVIDIA GPU
   # or: pip install -r requirements-torch-cpu.txt  # CPU-only machines
   pip install -r requirements.txt
   ```
4. **Model weights:** place `yolo11l-pose.pt` in `models/` (see *Model weights* below).
5. **Start the live pose tracker:**
   ```bash
   python -m mission_im_dancable
   ```
   Useful flags: `--camera 1`, `--width 640 --height 480`, `--device cpu`, `--model PATH`, `--conf 0.5`.
6. **Practice against a reference video:**
   ```bash
   python -m mission_im_dancable --source path/to/dance.mp4 --moves moves.json --mirror
   ```
   The first run extracts poses at `--sample-fps` (default 10) and caches them in `cache/`; later runs reuse the cache.
   **Preview the animated reference first:**
   ```bash
   python -m mission_im_dancable --preview --source path/to/dance.mp4 --moves moves.json --mirror
   ```
   **Play it as a rhythm game (pygame UI):**
   ```bash
   python -m mission_im_dancable --game --source path/to/dance.mp4 --moves moves.json --mirror
   ```
7. **Run tests:**
   ```bash
   python -m pytest tests/
   ```

### Practice mode

`--source VIDEO` enables move-by-move practice: the reference skeleton appears on the left, your live camera feed and skeleton on the right. The app loops inside each move's time range; press **N** for the next move, **P** for the previous one, **Q/Esc** to quit.

- **Supported video formats:** anything OpenCV/FFmpeg decodes — `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`. Local files only; keep them out of Git.
- **Match score:** your pose is normalized (hip-centred, torso-scaled) and compared against reference poses inside a `--window SECONDS` window (default ±0.15 s), so exact timestamp sync isn't required. Low-confidence landmarks are ignored rather than trusted; too few reliable landmarks shows `match -- (low confidence)`.
- **Corrections:** the single largest reliable mismatch is shown (e.g. `left wrist: move up-left`) and held stable so it doesn't flicker every frame.
- **`--mirror`:** flips the reference left-right so the instructor's left side maps to your right — the natural way to copy someone facing you.
- **Extraction cache:** stored under `cache/` (gitignored). A cache is reused only when the video file, sample rate, model, and confidence all match; otherwise it's re-extracted automatically.

**Move segmentation** is manual via a JSON dance-annotation file — `<video-stem>.moves.json` next to the video (auto-detected by the game):

```json
{
  "id": "my-dance",
  "title": "My dance",
  "difficulty": "medium",
  "moves": [
    {"name": "intro sway", "start": 0.0, "end": 2.4,
     "instruction": "shift weight left to right"},
    {"name": "step touch", "start": 2.4, "end": 6.0}
  ]
}
```

Timestamps are seconds; `start`/`end` must satisfy `0 <= start < end <= video duration`. `id`, `title`, `difficulty` and per-move `instruction` are all optional; instructions appear in Learn Mode. Write boundaries against the real video — the app never invents them. Without a moves file, the whole video is one move.

### Reference preview (`--preview`)

`--preview` opens a playback window before practice: the extracted pose sequence plays back as an animated skeleton at the original video speed — driven by elapsed wall time, not frame count, so speed is correct on any machine. Adjacent samples are interpolated for smooth motion; landmarks missing from a sample are never invented (a single confident endpoint is used, or the joint is skipped).

- **Controls:** SPACE pause/resume · ←/→ seek ±2 s · R restart · ENTER start practice · Q/Esc quit.
- **Layout:** reference skeleton on the left, your live camera skeleton on the right — same camera instance used by practice mode (no second capture). `--no-cam` runs the preview without a camera.
- **ENTER** continues straight into move-by-move practice with the same extraction, moves file, and camera.
- Person selection: the highest-confidence detected person is tracked per frame — the tracker does not switch between multiple dancers deliberately.

### Game mode (`--game`)

The pygame-ce dance-arcade UI: neon-purple theme with spotlights, equalizer, and a dance floor. `python -m mission_im_dancable --game` opens the home screen with two modes that share one dance library, one reference pipeline, and one pose engine:

- **LEARN** — study the dance at your own pace, no score and no fail state.
- **PLAY** — one continuous scored run through the whole dance.

#### Learn Mode

HOME → CHOOSE YOUR DANCE — LEARN (type a path, click a detected-video card, or BROWSE the filesystem — the browser shows a preview frame, duration/resolution, and SEGMENTS/POSES-CACHED badges for each video; ANALYZE extracts choreography + beats + audio in the background and reads `<stem>.moves.json` metadata: title, difficulty, per-move instructions) → the learn screen: reference skeleton on the left, live camera on the right, and a seekable segment timeline along the bottom.

- **Controls:** SPACE play/pause · ←/→ seek ±2 s · ,/. step one video frame · N/P next/previous segment (or click a segment block on the timeline) · L loop — off → selected segment → whole dance · -/= playback speed (0.25×–1×) · K mark the selected segment learning → practicing → mastered · E open the segment editor · M mirror · Q back. Buttons cover the same controls for mouse use.
- **Segment editor (E):** B splits the segment under the playhead, X merges with the next, [ ] nudges a boundary ±0.1 s, ↑/↓ selects, PgUp/PgDn reorders, N names the segment, ENTER preview-loops it, G generates a suggested draft (every-8th-beat cuts, or pose-motion stillness when no beat grid), A accepts suggestions, S saves to `<stem>.moves.json` (asks before overwriting). Suggested drafts render gold until accepted; saving reloads segments into every mode.
- **Feedback:** match % and the single biggest correction ("right shoulder: move up-left") — guidance only, never a score. Audio follows the playhead only at 1× speed.
- **Progress:** visiting a segment marks it *learning*; K cycles it through *practicing* and *mastered*. Marks are stored per dance in the user data dir — self-reported progress, never auto-judged from pose data.
- **Camera:** opens in the background; if it fails, Learn Mode keeps working as a watch-only viewer and R retries — the lesson never blocks on hardware.

#### Play Mode

HOME → CHOOSE YOUR DANCE — PLAY (same selection screen, same analyzed dance) → GET READY (camera opens in the background, live preview, "step into frame" detection hold, 3-2-1 countdown that pauses if tracking drops; camera errors get RETRY/BACK) → one continuous run from the first segment's start to the last segment's end (reference skeleton vs live skeleton + soundtrack re-anchored on drift, PERFECT/GREAT/MISS popups, combo milestones, particles, POSE LOST chip on tracking drops, per-segment progress strip — no seeking; a segment that ends without passing earns a 5-second COMPLETE THE MOVE grace loop, once per segment) → RESULTS (grade S/A/B/C/F, NEW PERSONAL BEST badge, per-move stats, retry / another dance / home). HIGH SCORES persists top runs locally; SETTINGS stores camera index, mirror default, fullscreen, match-window tolerance, and music volume in the user config dir.

Videos with no `.moves.json` sidecar still get playable steps: when a dance starts, boundaries are auto-drafted from the detected beat grid (or pose-motion stillness when there are no beats) and labeled "auto-segments" in Learn Mode — refine them in the editor (E) and save to make them real annotations.

- **Controls:** click or arrows/WS + ENTER on menus · SPACE/P pause · R restart the run · M mirror · Q/Esc back · F11 fullscreen · window is resizable.
- **Audio:** the selected video's audio track is decoded once (ffmpeg → cached WAV under `cache/`) and plays start-to-finish in Play Mode, follows the playhead in Learn Mode, and runs end-to-end in WATCH. Silent when the video has no audio track or no output device exists.
- **Scoring:** evaluations at fixed 0.25 s intervals (FPS-independent), 100 pts × combo multiplier (×1→×5 every 8 combo), misses break combo, duplicate frames never count, each eval bucketed into the segment owning its timestamp. Passing a move needs mean accuracy ≥ 60% + ≥ 50% reliable evals.
- **Reference display:** the extracted reference poses render as the same neon skeleton used everywhere else — synchronized to the video's timestamps, so pausing, seeking, and mirroring stay in sync.

### Model weights

`yolo11l-pose.pt` (Ultralytics YOLO11l-Pose) is required but **not committed to Git**. Download it into `models/`:

```bash
mkdir -p models
curl -L -o models/yolo11l-pose.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11l-pose.pt
# expected SHA-256: 61921abe1f2ed930bf28328a16b3162278cdb239a1e8280b6ac827119f13cee0
```

Or pass a custom location with `--model PATH`.

### Camera troubleshooting

- Default camera index is `0`; try `--camera 1` if your laptop enumerates IR/extra sensors first.
- `v4l2-ctl --list-devices` lists detected cameras on Linux.
- If another app is using the webcam (browser, Cheese, Zoom), close it first.
- The app exits with a clear error and code `2` when no camera is accessible — no fake frames are shown.

### Current status

- **Working features:** Live tracker; OpenCV practice and animated skeleton preview; pygame-ce dance arcade with timed pose scoring, combos, retries, results, beat-reactive effects.
- **Verified:** CUDA YOLO11l-Pose on RTX 3060, real webcam capture, extraction/cache on `dance_1.mp4`, per-sample timestamp sync and Pygame panel drawing. See the latest changelog entry for current automated test counts.
- **Known limitations:** manual move segmentation; game scoring against a real dancer remains unverified; interactive controls need an on-person check.
- **Libraries, APIs, and datasets used:** [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) (AGPL-3.0; also used for `yolo11l-pose.pt` weights and the `bus.jpg` test image), [PyTorch](https://pytorch.org), [OpenCV](https://opencv.org), [NumPy](https://numpy.org), [pygame-ce](https://pyga.me), [pytest](https://pytest.org).

See [AI_USAGE.md](AI_USAGE.md) for AI assistance and [CHANGELOG.md](CHANGELOG.md) for the development record.
