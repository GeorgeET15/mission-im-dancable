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
- **Technology stack:** Python 3.14, PyTorch (CUDA cu128), Ultralytics YOLO11n-Pose, OpenCV, NumPy, pygame-ce, pytest.

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
4. **Model weights:** place `yolo11n-pose.pt` in `models/` (see *Model weights* below).
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

**Move segmentation** is manual via a JSON file:

```json
{
  "moves": [
    {"name": "intro sway", "start": 0.0, "end": 2.4},
    {"name": "step touch", "start": 2.4, "end": 6.0}
  ]
}
```

Timestamps are seconds; `start`/`end` must satisfy `0 <= start < end <= video duration`. Without `--moves`, the whole video is one move.

### Reference preview (`--preview`)

`--preview` opens a playback window before practice: the extracted pose sequence plays back as an animated skeleton at the original video speed — driven by elapsed wall time, not frame count, so speed is correct on any machine. Adjacent samples are interpolated for smooth motion; landmarks missing from a sample are never invented (a single confident endpoint is used, or the joint is skipped).

- **Controls:** SPACE pause/resume · ←/→ seek ±2 s · R restart · ENTER start practice · Q/Esc quit.
- **Layout:** reference skeleton on the left, your live camera skeleton on the right — same camera instance used by practice mode (no second capture). `--no-cam` runs the preview without a camera.
- **ENTER** continues straight into move-by-move practice with the same extraction, moves file, and camera.
- Person selection: the highest-confidence detected person is tracked per frame — the tracker does not switch between multiple dancers deliberately.

### Game mode (`--game`)

The pygame-ce dance-arcade UI: neon-purple theme with spotlights, equalizer, and a dance floor. `python -m mission_im_dancable --game` opens the full flow:

- **Screens:** HOME → CHOOSE YOUR DANCE (type a path or pick a detected video, ANALYZE extracts choreography in the background, `<stem>.moves.json` auto-detected) → gameplay (reference Amy or skeleton vs live skeleton, PERFECT/GREAT/MISS popups, combo milestones, particles) → RESULTS (grade S/A/B/C/F, per-move stats, retry / another dance / home). HIGH SCORES persists top runs locally; SETTINGS stores camera index, mirror default, and Amy toggle in the user config dir.
- **Controls:** click or arrows/WS + ENTER on menus · SPACE confirm/start · R retry · P pause · M mirror · Q/Esc back · window is resizable.
- **Scoring:** unchanged from the GameSession — evaluations at fixed 0.25 s intervals (FPS-independent), 100 pts × combo multiplier (×1→×5 every 8 combo), misses break combo, duplicate frames never count. Passing needs mean accuracy ≥ 60% + ≥ 50% reliable evals.
- Native file dialog used if tkinter exists (it doesn't here) — fallback is the path field + detected-video buttons.

#### Amy reference animation

With `assets/blender/mission_dancer.blend` and Blender installed, **ANALYZE** uses the cached YOLO11 pose sequence to animate Amy's Mixamo rig and render one transparent PNG per reference sample. This happens in the selection screen's background worker, not the game loop. The first render may take a while; subsequent launches reuse video-specific renders under `cache/amy/`. No video, `.blend`, or rendered PNGs are tracked by Git. The older 48-frame T-pose/sway sprites are no longer used as a reference.

In the play screen, Amy's frame comes from the **reference video's timestamps and current session time**, so changing moves, pausing, retrying, and mirroring stay in sync. **Settings → AMY sprite reference** turns the character on or off. If Blender, the model, or a complete render is unavailable, the extracted animated skeleton remains the reference; pose scoring uses the original YOLO landmarks in either case.

This is a **2D camera-facing approximation** of the recorded motion: shoulders, elbows, wrists, torso/head direction, hips, knees, and ankles drive ten Mixamo bones plus limited root motion. A single-camera pose cannot determine depth, spins, hidden limbs, finger motion, or exact foot contact. Missing/low-confidence joints revert to rest rather than becoming invented moves. Amy does not follow the player's webcam; she follows the selected *reference video*. The render is an offline preprocessing step, not real-time 3D animation.

### Pose backend benchmark (`--benchmark-pose`)

Benchmark-only tooling compares **YOLO11n-Pose** (still the default backend everywhere) against **MediaPipe Pose Landmarker** on identical decoded frames. MediaPipe is *not* a runtime dependency — it lives in a separate venv so it can't break the game.

**Setup:**

```bash
python3 -m venv .bench-venv
.bench-venv/bin/pip install mediapipe opencv-python-headless numpy

mkdir -p models/mediapipe
curl -L -o models/mediapipe/pose_landmarker_heavy.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task
# expected SHA-256: 64437af838a65d18e5ba7a0d39b465540069bc8aae8308de3e318aad31fcbc7b
```

`full` and `lite` variants work too — swap the URL/filename and use `--pose-backends mediapipe-full` etc. Variants are never substituted for each other.

**Run (one backend per command is safest):**

```bash
source .venv/bin/activate
python -m mission_im_dancable --benchmark-pose --source dance_1.mp4 --pose-backends yolo
python -m mission_im_dancable --benchmark-pose --source dance_1.mp4 --pose-backends yolo-11l --max-frames 0
python -m mission_im_dancable --benchmark-pose --source dance_1.mp4 --pose-backends mediapipe-heavy
```

- `--max-frames N` caps decoded frames per backend (default **30** smoke test, `0` = whole video); the first 5 are warm-up and excluded from stats.
- Backends: `yolo` (= yolo11n-pose), `yolo-<variant>` (e.g. `yolo-11l`, `yolo-26m` → `models/yolo<variant>-pose.pt`), `mediapipe-<heavy|full|lite>`, optional `-gpu` suffix for the GL delegate. An explicit `--model PATH` overrides variant resolution.
- `--visualize-pose` writes an annotated skeleton overlay per backend to `output/pose_benchmark/visual_<backend>.mp4` (streamed, opt-in).
- Each backend runs in its **own worker process** — YOLO in `.venv`, MediaPipe in `.bench-venv` — strictly sequential, decoding the video one frame at a time. Results merge into `output/pose_benchmark/pose_benchmark.json` (+ `summary.txt`, per-backend `bench_*.json`, world-landmark `world_*.npz`). All under `output/` which is gitignored.

**Measured on this machine** (dance_1.mp4, 720×1280 @ 30 fps, 349 measured + 5 warm-up frames, conf 0.3, RTX 3060 Laptop CUDA):

| backend | weights | init s | mean ms | p95 ms | fps | det. rate | max gap | GPU MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| yolo (baseline) | yolo11n-pose.pt | 0.98 | 8.5 | 9.5 | 118 | 98.6% | 5 | 66 |
| yolo-11m | yolo11m-pose.pt | 1.04 | 11.0 | 12.6 | 91 | 99.1% | 1 | 173 |
| **yolo-11l** | yolo11l-pose.pt | 1.06 | 13.9 | 15.2 | 72 | **100%** | **0** | 202 |
| yolo-11x | yolo11x-pose.pt | 1.15 | 25.3 | 30.5 | 40 | **100%** | **0** | 359 |
| yolo-26m | yolo26m-pose.pt | 1.05 | 11.6 | 13.2 | 86 | 98.3% | 6 | 175 |
| yolo-26l | yolo26l-pose.pt | 1.08 | 14.0 | 15.5 | 71 | 98.6% | 5 | 200 |
| mediapipe-heavy | .task CPU (XNNPACK) | 0.47 | 46.9 | 50.6 | 21 | 94.8% | 17 | — |
| mediapipe-heavy-gpu | .task GL via EGL | 0.77 | 21.8 | 23.0 | 46 | 94.8% | 17 | — |

`yolo-<variant>` resolves `models/yolo<variant>-pose.pt`; weights from the official [ultralytics/assets v8.4.0 release](https://github.com/ultralytics/assets/releases/tag/v8.4.0). Every miss across every model is frames 5–10 — the dancer backlit by the centre spotlight inside stage haze. Only 11l/11x push through it; **YOLO26 gains nothing here** (26m is slightly worse than baseline). Recommendation: **yolo11l-pose is the best measured upgrade** — perfect detection and zero tracking gaps at ~72 FPS inference (P95 15 ms, well inside a 33 ms frame budget, 202 MB VRAM). Opt in per run with `--model models/yolo11l-pose.pt`; yolo11n stays the code default since scoring thresholds were tuned on its confidence distribution. YOLO11x adds nothing over 11l at 2× the latency.

**Important:** MediaPipe's GPU delegate initializes EGL against the **AMD Radeon iGPU** (Mesa radeonsi) on this hybrid laptop — it does *not* use the NVIDIA card. It is ~2.2× faster than its CPU delegate here but still ~2.5× slower than YOLO-on-CUDA. MediaPipe also emits 33-landmark world coordinates (metres, hip-centred **model estimates** — not mocap); saved to `world_*.npz` for future retargeting analysis. YOLO confidence and MediaPipe visibility/presence are different quantities — don't compare them numerically. Jitter numbers include the dancer's real motion.

Docs: [Pose Landmarker (Python)](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python) · [model variants](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker) · [Tasks API](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/PoseLandmarker)

### Model weights

`yolo11n-pose.pt` (Ultralytics YOLO11n-Pose) is required but **not committed to Git**. Download it into `models/`:

```bash
mkdir -p models
curl -L -o models/yolo11n-pose.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt
```

Or pass a custom location with `--model PATH`.

### Camera troubleshooting

- Default camera index is `0`; try `--camera 1` if your laptop enumerates IR/extra sensors first.
- `v4l2-ctl --list-devices` lists detected cameras on Linux.
- If another app is using the webcam (browser, Cheese, Zoom), close it first.
- The app exits with a clear error and code `2` when no camera is accessible — no fake frames are shown.

### Current status

- **Working features:** Live tracker; OpenCV practice and animated skeleton preview; pygame-ce dance arcade with timed pose scoring, combos, retries, results, beat-reactive effects and optional video-driven Amy reference.
- **Verified:** CUDA YOLO11n-Pose on RTX 3060, real webcam capture, extraction/cache on `dance_1.mp4`, headless Blender render of 118 Amy poses, per-sample timestamp sync and Pygame panel drawing. Pose benchmark ran on all 354 frames of `dance_1.mp4` for YOLO + MediaPipe Heavy (CPU and GL delegates). See the latest changelog entry for current automated test counts.
- **Known limitations:** Amy approximates visible 2D movement, not full 3D choreography; manual move segmentation; game scoring against a real dancer remains unverified; OpenCV preview/practice still show the skeleton; interactive controls need an on-person check.
- **Libraries, APIs, and datasets used:** [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) (AGPL-3.0; also used for `yolo11n-pose.pt` weights and the `bus.jpg` test image), [PyTorch](https://pytorch.org), [OpenCV](https://opencv.org), [NumPy](https://numpy.org), [pygame-ce](https://pyga.me), [pytest](https://pytest.org), [MediaPipe Tasks](https://github.com/google-ai-edge/mediapipe) (Apache-2.0; benchmark-only, isolated `.bench-venv`, `pose_landmarker_heavy.task` weights).

See [AI_USAGE.md](AI_USAGE.md) for AI assistance and [CHANGELOG.md](CHANGELOG.md) for the development record.
