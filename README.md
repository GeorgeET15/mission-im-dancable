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
6. **Run tests:**
   ```bash
   python -m pytest tests/
   ```

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

- **Working features:** milestone 1 live pose tracker — opens the camera, runs YOLO11n-Pose on CUDA (with CPU fallback via `--device cpu`), draws the detected skeleton and keypoints, shows FPS and inference time, and exits cleanly on Q/Escape. 21 unit tests cover config, geometry, timing, and rendering.
- **Verified:** torch 2.11.0+cu128, `torch.cuda.is_available() == True` on RTX 3060 Laptop GPU; YOLO11n-Pose loads and detects a person in a test image at ~11 ms/frame on CUDA; webcam reads real frames; the full app loop ran on the desktop display without errors.
- **Known limitations:** live tracker only — no reference choreography, scoring, or game UI yet. `pygame` proper has no Python 3.14 wheels, so `pygame-ce` is used. Skeleton quality on a moving dancer still needs an on-person check.
- **Libraries, APIs, and datasets used:** [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) (AGPL-3.0; also used for `yolo11n-pose.pt` weights and the `bus.jpg` test image), [PyTorch](https://pytorch.org), [OpenCV](https://opencv.org), [NumPy](https://numpy.org), [pygame-ce](https://pyga.me), [pytest](https://pytest.org).

See [AI_USAGE.md](AI_USAGE.md) for AI assistance and [CHANGELOG.md](CHANGELOG.md) for the development record.
