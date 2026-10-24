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

The pygame-ce mission-control interface turns each move into an objective:

- **Flow:** BRIEFING (move intro, SPACE to begin) → ACTIVE (perform the move once) → RESULT (passed → SPACE for next objective; failed → R to retry) → MISSION COMPLETE scorecard.
- **Controls:** SPACE/ENTER confirm · R retry (or replay after completion) · P pause · M toggle mirror · Q/Esc quit.
- **Scoring:** your pose is evaluated at fixed intervals (every 0.25 s of move time — points do **not** scale with frame rate). Each evaluation with accuracy ≥ 60% scores 100 pts × the combo multiplier (×1 rising to ×5 every 8 combo). A miss, low-confidence pose, or tracking loss breaks the combo. Duplicated camera frames never count as extra events.
- **Passing a move:** mean accuracy ≥ 60% **and** at least 50% of evaluations reliable — one lucky frame can't carry a move.
- **Feedback:** the hint panel shows the single largest reliable mismatch — a positional fix (`left wrist: move up-left`) or a joint-angle fix (`left elbow: straighten more`), held stable by hysteresis.
- Defaults live in `mission_im_dancable/game.py` (`GameConfig`); the OpenCV practice mode stays available by omitting `--game`.

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

- **Working features:** (1) live pose tracker — camera, CUDA YOLO11n-Pose, skeleton overlay, FPS/inference HUD, clean quit. (2) practice mode — `--source` + `--moves` gives looping move-by-move practice with match scoring and corrections (OpenCV). (3) game mode — `--game` runs the pygame mission-control UI: briefing/perform/result/scorecard flow, interval-based scoring, combo multiplier, pass/retry per move, joint-angle correction hints, pause and mirror toggle. (4) reference preview — `--preview` plays the animated reference skeleton at original speed (interpolated, elapsed-time driven) beside the live camera, with pause/seek/restart and ENTER to continue into practice.
- **Verified:** torch 2.11.0+cu128, `torch.cuda.is_available() == True` on RTX 3060 Laptop GPU; YOLO11n-Pose ~11 ms/frame on CUDA; webcam reads real frames; reference extraction + cache-hit verified on a synthetic 6 s clip (60/60 detections); tracker, practice, game, and preview (with and without camera) all ran end-to-end on the desktop display; 95 unit tests pass.
- **Known limitations:** reference shows as a skeleton, not the decoded video frames; move segmentation is manual; game-mode scoring verified by simulation tests, but a full run against a real dancing person is untested; preview key controls need an interactive keypress check. `pygame-ce` used instead of `pygame` (no Python 3.14 wheels).
- **Libraries, APIs, and datasets used:** [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) (AGPL-3.0; also used for `yolo11n-pose.pt` weights and the `bus.jpg` test image), [PyTorch](https://pytorch.org), [OpenCV](https://opencv.org), [NumPy](https://numpy.org), [pygame-ce](https://pyga.me), [pytest](https://pytest.org).

See [AI_USAGE.md](AI_USAGE.md) for AI assistance and [CHANGELOG.md](CHANGELOG.md) for the development record.
