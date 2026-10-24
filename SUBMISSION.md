# Project submission

Complete this file during the hackathon and use it to prepare your submission through the method announced by the organizers. Do not submit your project to the template repository.

## Project

- **Project name:** Mission Im-Dancable
- **Short description:** An AI-powered dance-teaching rhythm game. Pose estimation tracks your body through the webcam, teaches choreography one move at a time, and scores how closely you match the reference — for anyone who wants to learn to dance without a studio.
- **Code repository URL:** https://github.com/GeorgeET15/mission-im-dancable
- **Final submitted implementation commit SHA:** TODO — record at submission time
- **Working demo URL or local demo steps:** Local demo — `python -m mission_im_dancable` (see README for setup).
- **Demo access requirements:** Webcam; NVIDIA GPU recommended but CPU fallback works. No credentials required.

## Participants

Participate solo or in a team of up to 4. Remove unused rows.

| Participant name | GitHub handle | Contribution |
| --- | --- | --- |
| TODO | GeorgeET15 | Project lead — implementation |

## Run and demonstrate

- **Prerequisites and versions:** Python 3.14, webcam; NVIDIA GPU with CUDA-12.8-capable driver recommended (verified: RTX 3060 Laptop, driver 595.91.07).
- **Install command:** `pip install -r requirements-torch-cuda.txt && pip install -r requirements.txt` (use `requirements-torch-cpu.txt` on CPU-only machines).
- **Required environment variable names:** None.
- **Start command:** `python -m mission_im_dancable`
- **Demo walkthrough:** Run the start command; the camera feed opens with a skeleton overlay on the detected person, FPS and inference-time HUD. Press Q or Escape to quit. (Full choreography demo pending later milestones.)
- **Checks performed and results:** CUDA available and GPU-named correctly; YOLO11n-Pose inference on test image at ~11 ms; 21/21 unit tests pass; live camera loop pending an on-person test.
- **Known limitations or incomplete features:** Milestone 1 only — live tracker, no scoring/game loop yet. `pygame-ce` used instead of `pygame` (no Python 3.14 wheels).
- **Libraries, APIs, and public datasets used:** Ultralytics YOLO11 (AGPL-3.0) + `yolo11n-pose.pt` weights + `bus.jpg` sample image; PyTorch cu128; OpenCV; NumPy; pygame-ce; pytest.
- **AI usage record:** [AI_USAGE.md](AI_USAGE.md)
- **Development change log:** [CHANGELOG.md](CHANGELOG.md)

## Submission checklist

- [ ] The project implementation started from scratch at the event.
- [ ] I participated solo or with no more than 4 team members in total.
- [ ] Codex was used during development and its usage is documented.
- [ ] The code repository URL, short description, and working demo are provided.
- [ ] README setup and demo instructions match the submitted code.
- [ ] Actual checks and known limitations are documented.
- [ ] Meaningful changes are documented and committed throughout development.
- [ ] The final implementation is committed and pushed, and its SHA is recorded above.
- [ ] No credentials or private data are included in the repository or demo instructions.
- [ ] The organizers' announced submission method and timing have been followed.

Event reference: [DevDay Exchange Community: Kochi on Luma](https://luma.com/z10nz8gp). The event wraps up by **10:00 AM IST on 11 October 2026**.
