from pathlib import Path

import numpy as np
import pytest

from mission_im_dancable.moves import Move
from mission_im_dancable.practice import PracticeSession, _scaled_panel_pose
from mission_im_dancable.reference import ReferenceChoreography


def _choreo(duration=10.0, sample_fps=10.0) -> ReferenceChoreography:
    n = int(duration * sample_fps)
    kpts = np.zeros((n, 17, 3), dtype=np.float32)
    kpts[:, :, 2] = 0.9
    for i in range(n):
        kpts[i, 0, 0] = i  # distinguish samples by nose x position
    return ReferenceChoreography(
        video_path=Path("clip.mp4"),
        video_fps=30.0,
        frame_count=int(duration * 30),
        width=640,
        height=480,
        sample_fps=sample_fps,
        timestamps=np.arange(n) / sample_fps,
        keypoints=kpts,
        model_name="m.pt",
    )


def _session():
    moves = [Move("a", 0.0, 2.0), Move("b", 3.0, 6.0), Move("c", 8.0, 10.0)]
    return PracticeSession(_choreo(), moves, window_s=0.15)


def test_session_starts_at_first_move():
    s = _session()
    assert s.move_index == 0 and s.time == 0.0 and s.move.name == "a"


def test_advance_loops_inside_move():
    s = _session()
    s.advance(2.6)  # past move a's end (2.0) → wraps to 0.6
    assert s.time == pytest.approx(0.6)
    assert s.move_index == 0


def test_set_move_clamps_and_resets_time():
    s = _session()
    s.set_move(2)
    assert s.move_index == 2 and s.time == pytest.approx(8.0)
    s.set_move(99)
    assert s.move_index == 2
    s.set_move(-3)
    assert s.move_index == 0


def test_window_keypoints_selects_nearby_samples():
    s = _session()
    s.advance(1.0)
    window = s.window_keypoints()
    # 10 fps sampling, ±0.15s → about 4 samples around t=1.0
    ts = np.arange(window.shape[0])  # just verify shape/content came back
    assert 2 <= window.shape[0] <= 5
    assert all(r[:, 2].max() > 0 for r in window)


def test_nearest_keypoints_picks_closest_sample():
    s = _session()
    s.advance(1.23)
    nearest = s.nearest_keypoints()
    assert nearest is not None
    # sample at t=1.2 has nose x = 12
    assert nearest[0, 0] == pytest.approx(12)


def test_nearest_keypoints_empty_choreography():
    choreo = _choreo()
    choreo.timestamps = np.array([])
    choreo.keypoints = np.zeros((0, 17, 3), dtype=np.float32)
    s = PracticeSession(choreo, [Move("a", 0, 1)], 0.15)
    assert s.nearest_keypoints() is None
    assert s.window_keypoints().shape == (0, 17, 3)


def test_session_requires_moves():
    with pytest.raises(ValueError):
        PracticeSession(_choreo(), [], 0.15)


def test_scaled_panel_pose_fits_and_preserves_conf():
    kpts = np.zeros((17, 3))
    kpts[:, 0] = 640  # all at right edge of a 640x480 video
    kpts[:, 2] = 0.9
    out = _scaled_panel_pose(kpts, src_w=640, src_h=480, dst_w=320, dst_h=240)
    assert out[:, 0].max() <= 320
    assert out[:, 1].max() <= 240
    assert np.allclose(out[:, 2], 0.9)
