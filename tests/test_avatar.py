"""Headless pose-to-rig target and cache tests (no Blender/display needed)."""

from pathlib import Path

import numpy as np
import pytest

from mission_im_dancable.avatar import (
    BONES, animation_targets, cache_path, frame_manifest, pose_targets,
    render_choreography,
)
from mission_im_dancable.reference import ReferenceChoreography
from mission_im_dancable.ui.sprite import SpriteAnim, load_frames


def pose():
    k = np.zeros((17, 3), dtype=np.float32)
    joints = {
        0: (50, 12), 5: (65, 40), 6: (35, 40),
        7: (78, 40), 8: (22, 40), 9: (90, 40), 10: (10, 40),
        11: (60, 80), 12: (40, 80),
        13: (60, 105), 14: (40, 105), 15: (60, 130), 16: (40, 130),
    }
    for i, (x, y) in joints.items():
        k[i] = (x, y, .9)
    return k


def choreography(keypoints):
    frames = np.asarray(keypoints, dtype=np.float32)
    return ReferenceChoreography(
        video_path=Path("dance.mp4"), video_fps=30, frame_count=30,
        width=100, height=150, sample_fps=10,
        timestamps=np.arange(len(frames)) / 10, keypoints=frames,
        model_name="fake",
    )


def test_arm_and_leg_directions_follow_visible_pose():
    raised = pose()
    raised[7, :2] = (65, 20)  # elbow directly above shoulder
    raised[9, :2] = (65, 3)   # wrist above elbow
    raised[13, :2] = (80, 80)  # knee to the right of hip
    directions = pose_targets(raised)
    assert directions[BONES.index("LeftArm")] == pytest.approx([0, 0, 1])
    assert directions[BONES.index("LeftForeArm")] == pytest.approx([0, 0, 1])
    assert directions[BONES.index("LeftUpLeg")] == pytest.approx([1, 0, 0])
    assert pose_targets(pose())[BONES.index("RightArm")] == pytest.approx([-1, 0, 0])


def test_missing_unreliable_or_zero_length_landmarks_do_not_drive_bones():
    p = pose()
    p[7, 2] = 0.1
    p[14, :2] = p[12, :2]
    targets = pose_targets(p, confidence=.35)
    assert np.isnan(targets[BONES.index("LeftArm")]).all()
    assert np.isnan(targets[BONES.index("LeftForeArm")]).all()
    assert np.isnan(targets[BONES.index("RightUpLeg")]).all()
    assert np.isfinite(targets[BONES.index("RightArm")]).all()


def test_targets_keep_video_order_and_root_relative_to_first_valid_pose():
    a = pose()
    b = pose()
    b[:, 0] += 10
    directions, roots = animation_targets(choreography([a, b]), .35)
    assert directions.shape == (2, len(BONES), 3)
    assert roots[0] == pytest.approx([0, 0])
    assert roots[1, 0] > 0
    assert roots[1, 1] == 0


def test_cache_key_responds_to_pose_model_and_confidence(tmp_path):
    model = tmp_path / "amy.blend"
    model.write_bytes(b"rig")
    a = choreography([pose()])
    first = cache_path(a, tmp_path, model, .35)
    assert first != cache_path(a, tmp_path, model, .5)
    a.keypoints[0, 5, 0] += 5
    assert first != cache_path(a, tmp_path, model, .35)
    model.write_bytes(b"new rig")
    assert first != cache_path(a, tmp_path, model, .35)


def test_valid_cached_render_works_without_blender(tmp_path, monkeypatch):
    import json

    model = tmp_path / "amy.blend"
    model.write_bytes(b"rig")
    sample = choreography([pose(), pose()])
    cached = cache_path(sample, tmp_path, model, .35)
    cached.mkdir(parents=True)
    for i in (1, 2):
        (cached / f"frame_{i:04d}.png").write_bytes(b"png")
    (cached / "manifest.json").write_text(
        json.dumps({"version": 1, "timestamps": [0, .1]})
    )
    monkeypatch.setattr("mission_im_dancable.avatar.shutil.which", lambda _: None)
    assert render_choreography(sample, tmp_path, model=model) == cached


def test_failed_renderer_leaves_no_valid_manifest(tmp_path, monkeypatch):
    model = tmp_path / "amy.blend"
    model.write_bytes(b"rig")
    monkeypatch.setattr("mission_im_dancable.avatar.subprocess.run",
                        lambda *a, **k: type("Result", (), {"returncode": 1, "stderr": "render failed"})())
    with pytest.raises(RuntimeError, match="render failed"):
        render_choreography(choreography([pose()]), tmp_path, model=model, blender="blender")
    assert not list(tmp_path.rglob("manifest.json"))


def test_missing_blender_or_model_raises_for_skeleton_fallback(tmp_path):
    with pytest.raises(RuntimeError, match="Amy model"):
        render_choreography(choreography([pose()]), tmp_path, model=tmp_path / "missing.blend")


def test_timestamped_sprite_seeks_and_clamps_instead_of_looping():
    anim = SpriteAnim(["a", "b", "c"], timestamps=[0.0, .4, 1.2])
    assert [anim.surface_at(t) for t in (0, .3, .4, 1.1, 1.2, 99)] == [
        "a", "a", "b", "b", "c", "c",
    ]
    assert anim.surface_at(.4) == anim.surface_at(.4)  # paused clock
    assert anim.surface_at(.1) == "a"  # seek back
    with pytest.raises(ValueError, match="increase"):
        SpriteAnim(["a", "b"], timestamps=[0, 0])


def test_incomplete_frame_manifest_is_rejected(tmp_path):
    (tmp_path / "manifest.json").write_text('{"version": 1, "timestamps": [0.0, 0.1]}')
    assert frame_manifest(tmp_path, 2) is None
    import pygame
    assert load_frames(pygame, tmp_path) is None


def test_play_panel_uses_session_time_and_falls_back_to_skeleton(monkeypatch):
    import pygame
    import mission_im_dancable.ui.screens as screens
    from mission_im_dancable.game import GameSession
    from mission_im_dancable.moves import Move

    pygame.font.init()
    surf = pygame.Surface((1280, 720))
    calls = []
    ctx = type("Context", (), {})()
    ctx.choreo = choreography([pose(), pose()])
    ctx.mirror = True
    ctx.cfg = type("Config", (), {"confidence": .35})()
    ctx.settings = {"amy": True}
    ctx.fonts = {"sm": pygame.font.Font(None, 20)}
    screen = screens.PlayScreen()
    screen.enter()
    screen.session = GameSession([Move("move", 0, 1)])
    screen.session.time = .1

    class FakeAmy:
        def draw(self, _pygame, _surface, rect, t, mirror=False):
            calls.append((rect, t, mirror))

    ctx.amy = FakeAmy()
    screen._panels(ctx, surf)
    assert calls == [((30, 110, 600, 470), .1, True)]
    ctx.amy = None
    skeleton_calls = []
    monkeypatch.setattr(screens, "_neon_skeleton", lambda *args: skeleton_calls.append(args))
    screen._panels(ctx, surf)
    assert len(skeleton_calls) == 1


def test_selection_handles_avatar_render_failure_without_blocking_game(tmp_path, monkeypatch):
    import mission_im_dancable.ui.screens as screens
    from mission_im_dancable.config import AppConfig

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    monkeypatch.setattr(screens, "_probe_video", lambda _: (1.0, 100, 150))
    monkeypatch.setattr(screens, "load_or_extract", lambda *a, **k: choreography([pose()]))
    monkeypatch.setattr(screens, "load_or_detect_beats", lambda *a: None)
    monkeypatch.setattr(screens, "render_choreography",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("Blender absent")))
    ctx = type("Context", (), {})()
    ctx.cfg = AppConfig()
    ctx.settings = {"amy": True}
    ctx.tracker = object()  # avoid loading a real model
    ctx.cache_dir = tmp_path
    ctx.video_path = None
    ctx.moves_path = None
    ctx.choreo = None
    ctx.moves = None
    ctx.amy = None
    ctx.avatar_dir = None
    ctx.mode = "performance"
    ctx.mirror = False
    screen = screens.SelectScreen()
    screen.enter()
    screen.field.text = str(video)
    screen._analyze(ctx)
    screen._thread.join(timeout=5)
    assert not screen._thread.is_alive()
    screen.update(ctx, .016)
    assert screen.btn_start.enabled
    assert ctx.amy is None
    assert "using skeleton" in screen.status
