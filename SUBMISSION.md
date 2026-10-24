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
- **Demo walkthrough:** `python -m mission_im_dancable --game` — home screen: **LEARN** (study the dance — seek, slow-mo, segment loop, pose match feedback, self-marked progress) or **PLAY** (camera check → countdown → one continuous scored run → results + grade). Pick a detected video or BROWSE/ANALYZE any clip; `dance_1.moves.json` ships real annotated segments. Fallbacks: `--preview --source X` (animated reference), `--source X` (OpenCV practice), no args (live tracker).
- **Checks performed and results:** CUDA verified on RTX 3060; YOLO11l-Pose ~14 ms/frame (~72 FPS); 195/195 unit tests pass; extraction + caching verified on real clip `dance_1.mp4`; headless end-to-end smoke drove the full LEARN and PLAY flows (screenshots reviewed); camera-open and camera-failure paths exercised. A full game run against a real dancer is untested.
- **Known limitations or incomplete features:** Real-dancer playthrough unverified; reference shown as skeleton rather than video pixels; move segmentation is manual (hand-written `.moves.json` per video, format in README); no audio at slowed Learn playback; `pygame-ce` used instead of `pygame` (no Python 3.14 wheels).
- **Libraries, APIs, and public datasets used:** Ultralytics YOLO11 (AGPL-3.0) + `yolo11l-pose.pt` weights + `bus.jpg` sample image; PyTorch cu128; OpenCV; NumPy; pygame-ce; pytest.
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
