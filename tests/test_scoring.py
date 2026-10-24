import numpy as np
import pytest

from mission_im_dancable.scoring import (
    CorrectionTracker,
    ScoreSmoother,
    best_window_score,
    describe_correction,
    mirror_keypoints,
    score_pose,
    worst_mismatch,
)


def _standing_pose(scale=1.0, offset=(0.0, 0.0), conf=0.9) -> np.ndarray:
    """T-pose-ish synthetic person in pixel coords."""
    xy = np.array(
        [
            [0, -2], [-0.1, -2.1], [0.1, -2.1], [-0.3, -2.0], [0.3, -2.0],
            [-1, -1], [1, -1],      # shoulders
            [-2, -1], [2, -1],      # elbows
            [-3, -1], [3, -1],      # wrists
            [-0.5, 0], [0.5, 0],    # hips
            [-0.5, 1], [0.5, 1],    # knees
            [-0.5, 2], [0.5, 2],    # ankles
        ],
        dtype=np.float64,
    )
    kpts = np.zeros((17, 3))
    kpts[:, :2] = xy * scale + np.asarray(offset)
    kpts[:, 2] = conf
    return kpts


def test_identical_pose_scores_one():
    pose = _standing_pose()
    result = score_pose(pose, pose.copy(), conf_threshold=0.3)
    assert result is not None
    assert result.score == pytest.approx(1.0)
    assert result.reliable_count == 17
    assert worst_mismatch(result) is None


def test_score_invariant_to_position_and_size():
    a = _standing_pose()
    b = _standing_pose(scale=2.7, offset=(400, 300))
    result = score_pose(a, b, conf_threshold=0.3)
    assert result is not None and result.score > 0.99


def test_moved_keypoint_lowers_score_and_is_flagged():
    ref = _standing_pose()
    live = _standing_pose()
    live[9, :2] += np.array([0.8, -1.2])  # left wrist raised high
    result = score_pose(live, ref, conf_threshold=0.3)
    assert result is not None and result.score < 0.95
    assert worst_mismatch(result) == 9
    assert "wrist" in describe_correction(9, live, ref)


def test_correction_direction_points_to_reference():
    ref = _standing_pose()
    live = _standing_pose()
    live[9, :2] += np.array([0.9, 0.9])  # wrist too low and too far right
    text = describe_correction(9, live, ref)
    assert "up" in text and "left" in text


def test_low_confidence_points_are_ignored_not_trusted():
    ref = _standing_pose()
    live = _standing_pose()
    live[9, :2] = np.array([9999.0, 9999.0])  # wild position...
    live[9, 2] = 0.01                        # ...but untrusted confidence
    result = score_pose(live, ref, conf_threshold=0.3)
    assert result is not None
    assert result.reliable_count == 16
    assert result.score > 0.99          # wild point did not tank the score
    assert np.isnan(result.keypoint_errors[9])


def test_too_few_reliable_points_returns_none():
    live = _standing_pose()
    live[:, 2] = 0.0  # nothing reliable
    assert score_pose(live, _standing_pose(), conf_threshold=0.3) is None


def test_mirror_swaps_left_right_and_flips_x():
    kpts = np.zeros((17, 3))
    kpts[9] = [10, 20, 0.9]   # left wrist
    kpts[10] = [30, 20, 0.9]  # right wrist
    mirrored = mirror_keypoints(kpts, axis_width=100)
    assert mirrored[9, 0] == pytest.approx(70)   # was right wrist at x=30
    assert mirrored[10, 0] == pytest.approx(90)  # was left wrist at x=10


def test_mirrored_reference_matches_mirrored_user():
    # Instructor raises THEIR left wrist; a mirroring user raises their right.
    ref = _standing_pose()
    ref[9, :2] += np.array([0.0, -1.0])
    live = _standing_pose()
    live[10, :2] += np.array([0.0, -1.0])  # user's right wrist up

    ref_m = mirror_keypoints(ref, axis_width=0)  # coords centred near 0
    plain = score_pose(live, ref, 0.3)
    mirrored = score_pose(live, ref_m, 0.3)
    assert plain is not None and mirrored is not None
    assert mirrored.score > plain.score
    # mirrored ref expects the moved wrist at idx 10 → both wrists reliable
    assert mirrored.reliable_count == 17


def test_best_window_score_skips_missing_detections():
    live = _standing_pose()
    window = np.stack(
        [np.zeros((17, 3)), _standing_pose(), _standing_pose()]
    )
    window[2][9, :2] += np.array([5.0, 5.0])  # third candidate is a worse match
    best = best_window_score(live, window, conf_threshold=0.3)
    assert best is not None
    assert best[1] == 1  # index of the best-matching reference row
    assert best[0].score > 0.99


def test_best_window_score_all_missing():
    live = _standing_pose()
    window = np.zeros((3, 17, 3))
    assert best_window_score(live, window, 0.3) is None


def test_score_smoother():
    s = ScoreSmoother(alpha=0.5)
    assert s.update(1.0) == pytest.approx(1.0)
    assert s.update(0.0) == pytest.approx(0.5)
    assert s.update(None) == pytest.approx(0.5)  # keeps last value


def test_correction_tracker_holds_and_switches():
    tracker = CorrectionTracker(hold_frames=3)
    assert tracker.update(5) == 5          # first mismatch becomes current
    assert tracker.update(7) == 5          # flicker doesn't switch
    assert tracker.update(7) == 5
    assert tracker.update(7) == 7          # persistent mismatch wins
    assert tracker.update(None) == 7       # missing data keeps current
    assert tracker.update(5) == 7
