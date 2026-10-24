import numpy as np
import pytest

from mission_im_dancable.geometry import (
    KEYPOINT_NAMES,
    SKELETON_EDGES,
    joint_angle,
    normalize_keypoints,
)


def test_keypoint_table_shape():
    assert len(KEYPOINT_NAMES) == 17
    assert all(0 <= a < 17 and 0 <= b < 17 for a, b in SKELETON_EDGES)


def test_joint_angle_right_angle():
    angle = joint_angle(np.array([1.0, 0.0]), np.array([0.0, 0.0]), np.array([0.0, 1.0]))
    assert angle == pytest.approx(90.0)


def test_joint_angle_straight_line():
    angle = joint_angle(np.array([-1.0, 0.0]), np.array([0.0, 0.0]), np.array([1.0, 0.0]))
    assert angle == pytest.approx(180.0)


def test_joint_angle_degenerate_returns_nan():
    assert np.isnan(joint_angle(np.array([0.0, 0.0]), np.array([0.0, 0.0]), np.array([1.0, 0.0])))


def _synthetic_pose() -> np.ndarray:
    kpts = np.zeros((17, 3))
    kpts[5] = [50, 100, 0.9]   # left shoulder
    kpts[6] = [150, 100, 0.9]  # right shoulder
    kpts[11] = [60, 200, 0.9]  # left hip
    kpts[12] = [140, 200, 0.9] # right hip
    kpts[0] = [100, 40, 0.9]   # nose
    return kpts


def test_normalize_keypoints_centres_mid_hip():
    norm = _synthetic_pose()
    norm = normalize_keypoints(norm)
    mid_hip = (norm[11, :2] + norm[12, :2]) / 2.0
    assert np.allclose(mid_hip, [0.0, 0.0])


def test_normalize_keypoints_preserves_confidence():
    kpts = _synthetic_pose()
    norm = normalize_keypoints(kpts)
    assert np.allclose(norm[:, 2], kpts[:, 2])


def test_normalize_keypoints_zero_scale_is_safe():
    kpts = np.zeros((17, 3))
    out = normalize_keypoints(kpts)
    assert np.isfinite(out).all()
