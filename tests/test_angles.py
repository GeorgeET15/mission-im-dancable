import numpy as np
import pytest

from mission_im_dancable.scoring import (
    JOINTS,
    analyze_pose,
    angle_errors,
    describe_angle_correction,
    joint_angle_deg,
    mirror_keypoints,
)


def _pose(conf=0.9) -> np.ndarray:
    """T-pose with straight limbs (all joint angles ~180°)."""
    xy = np.array(
        [
            [0, -2], [-0.1, -2.1], [0.1, -2.1], [-0.3, -2.0], [0.3, -2.0],
            [-1, -1], [1, -1],
            [-2, -1], [2, -1],
            [-3, -1], [3, -1],
            [-0.5, 0], [0.5, 0],
            [-0.5, 1], [0.5, 1],
            [-0.5, 2], [0.5, 2],
        ],
        dtype=np.float64,
    )
    kpts = np.zeros((17, 3))
    kpts[:, :2] = xy
    kpts[:, 2] = conf
    return kpts


def test_angle_errors_zero_for_identical_poses():
    errors = angle_errors(_pose(), _pose(), 0.3)
    assert errors  # joints were measured
    assert all(e < 1e-6 for e in errors.values())


def test_bent_elbow_reports_degrees():
    ref = _pose()
    live = _pose()
    live[9, :2] = np.array([-2.0, 0.0])  # left wrist pulled toward shoulder
    errors = angle_errors(live, ref, 0.3)
    assert "left elbow" in errors
    assert errors["left elbow"] == pytest.approx(90.0, abs=1.0)


def test_low_confidence_joint_is_omitted():
    ref = _pose()
    live = _pose()
    live[9, 2] = 0.01  # left wrist untrusted → left elbow unmeasurable
    errors = angle_errors(live, ref, 0.3)
    assert "left elbow" not in errors
    assert "right elbow" in errors


def test_degenerate_joint_is_omitted():
    live = _pose()
    live[7, :2] = live[5, :2]  # elbow stacked on shoulder → zero-length limb
    errors = angle_errors(live, _pose(), 0.3)
    assert "left elbow" not in errors


def test_analyze_pose_combines_position_and_angles():
    ref = _pose()
    live = _pose()
    live[9, :2] = np.array([-2.0, 0.0])
    analysis = analyze_pose(live, ref, 0.3)
    assert analysis is not None
    worst = analysis.worst_angle()
    assert worst is not None and worst[0] == "left elbow"
    assert analysis.pose_score.score < 1.0


def test_analyze_pose_none_when_unreliable():
    live = _pose()
    live[:, 2] = 0.0
    assert analyze_pose(live, _pose(), 0.3) is None


def test_worst_angle_below_threshold_is_none():
    ref = _pose()
    live = _pose()
    live[9, :2] += np.array([0.05, 0.05])  # tiny wobble, ~few degrees
    analysis = analyze_pose(live, ref, 0.3)
    assert analysis is not None
    assert analysis.worst_angle() is None


def test_angles_are_mirror_invariant():
    ref = _pose()
    ref[9, :2] = np.array([-2.0, 0.0])  # bent LEFT elbow
    live = _pose()
    live[10, :2] = np.array([2.0, 0.0])  # user bends RIGHT elbow (mirroring)
    errors = angle_errors(live, mirror_keypoints(ref, axis_width=0), 0.3)
    assert errors["right elbow"] < 5.0  # mirrored ref matches user's right elbow


def test_describe_angle_correction_direction():
    assert "straighten" in describe_angle_correction("left elbow", 90.0, 160.0)
    assert "bend" in describe_angle_correction("left elbow", 170.0, 90.0)


def test_joint_angle_deg_lookup():
    pose = _pose()
    deg = joint_angle_deg(pose, "left elbow")
    assert deg == pytest.approx(180.0, abs=1.0)
    assert joint_angle_deg(pose, "not a joint") is None


def test_joint_table_uses_valid_indices():
    assert all(0 <= i < 17 for _, a, b, c in JOINTS for i in (a, b, c))
