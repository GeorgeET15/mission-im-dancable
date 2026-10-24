from pathlib import Path

import numpy as np
import pytest

from mission_im_dancable.playback import PlaybackClock
from mission_im_dancable.preview import reference_panel
from mission_im_dancable.reference import ReferenceChoreography, pose_at


def _choreo(duration=2.0, sample_fps=10.0) -> ReferenceChoreography:
    n = int(duration * sample_fps)
    kpts = np.zeros((n, 17, 3), dtype=np.float32)
    kpts[:, :, 2] = 0.9
    for i in range(n):
        kpts[i, 0, 0] = float(i)  # nose x = sample index
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


# --- PlaybackClock ---------------------------------------------------------


def test_clock_advances_by_elapsed_wall_time():
    c = PlaybackClock(10.0)
    c.now(wall=100.0)
    assert c.now(wall=100.5) == pytest.approx(0.5)
    assert c.now(wall=101.25) == pytest.approx(1.25)


def test_clock_pause_and_resume():
    c = PlaybackClock(10.0)
    c.now(0.0)
    c.now(1.0)
    c.pause()
    assert c.now(5.0) == pytest.approx(1.0)  # frozen while paused
    c.play()
    assert c.now(6.0) == pytest.approx(1.0)  # first tick after resume, no jump
    assert c.now(7.5) == pytest.approx(2.5)


def test_clock_seek_clamps_to_bounds():
    c = PlaybackClock(10.0)
    c.seek(5.0)
    assert c.time == 5.0
    c.seek(-3.0)
    assert c.time == 0.0
    c.seek(99.0)
    assert c.time == 10.0


def test_clock_skip_and_restart():
    c = PlaybackClock(10.0)
    c.seek(4.0)
    c.skip(2.0)
    assert c.time == 6.0
    c.skip(-10.0)
    assert c.time == 0.0
    c.restart()
    assert c.time == 0.0 and c.playing


def test_clock_stops_at_end():
    c = PlaybackClock(1.0)
    c.now(0.0)
    t = c.now(5.0)
    assert t == 1.0
    assert not c.playing
    assert c.finished


def test_clock_rejects_bad_duration():
    with pytest.raises(ValueError):
        PlaybackClock(0.0)


def test_clock_playback_speed_independent_of_tick_rate():
    for step in (0.016, 0.05, 0.2):
        c = PlaybackClock(10.0)
        wall = 0.0
        c.now(wall)
        for _ in range(10):
            wall += step
            c.now(wall)
        assert c.time == pytest.approx(10 * step)


# --- pose_at interpolation --------------------------------------------------


def test_pose_at_exact_sample():
    c = _choreo()
    pose = pose_at(c, 0.5)
    assert pose[0, 0] == pytest.approx(5.0)  # sample 5 at t=0.5


def test_pose_at_interpolates_midpoint():
    c = _choreo()
    pose = pose_at(c, 0.55)  # halfway between samples 5 and 6
    assert pose[0, 0] == pytest.approx(5.5)


def test_pose_at_clamps_out_of_range():
    c = _choreo()
    assert pose_at(c, -1.0)[0, 0] == 0.0     # first sample
    assert pose_at(c, 99.0)[0, 0] == 19.0    # last sample


def test_pose_at_handles_missing_landmarks():
    c = _choreo()
    c.keypoints[6][3, 2] = 0.0  # left ear missing in sample 6
    mid = pose_at(c, 0.55)      # between samples 5 (present) and 6 (missing)
    # falls back to the present endpoint's full confidence rather than
    # interpolating toward a phantom landmark
    assert mid[3, 2] == pytest.approx(0.9)


def test_pose_at_both_missing_stays_zero():
    c = _choreo()
    c.keypoints[5][3, 2] = 0.0
    c.keypoints[6][3, 2] = 0.0
    mid = pose_at(c, 0.55)
    assert mid[3, 2] == 0.0


def test_pose_at_empty_choreography():
    c = _choreo()
    c.timestamps = np.array([])
    c.keypoints = np.zeros((0, 17, 3), dtype=np.float32)
    assert pose_at(c, 1.0) is None


# --- panel rendering --------------------------------------------------------


def test_reference_panel_draws_skeleton():
    c = _choreo()
    c.keypoints[:, :, :2] = 320.0  # put all joints mid-panel
    pose = pose_at(c, 1.0)
    img = reference_panel(c, pose, 1.0, True, 640, 480, 0.3, False)
    assert img.shape == (480, 640, 3)
    assert img.sum() > 0


def test_reference_panel_no_pose_still_renders():
    c = _choreo()
    empty = np.zeros((17, 3), dtype=np.float32)
    img = reference_panel(c, empty, 1.0, True, 640, 480, 0.3, False)
    assert img.sum() > 0  # text + progress bar still drawn


def test_reference_panel_mirrors():
    c = _choreo()
    pose = pose_at(c, 1.0)
    img_plain = reference_panel(c, pose, 1.0, True, 640, 480, 0.3, False)
    img_mirror = reference_panel(c, pose, 1.0, True, 640, 480, 0.3, True)
    assert not np.array_equal(img_plain, img_mirror)
