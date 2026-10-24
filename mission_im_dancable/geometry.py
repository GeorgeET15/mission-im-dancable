"""Landmark math used for rendering and future pose scoring.

COCO keypoint indices used by YOLO pose models.
"""

from __future__ import annotations

import numpy as np

# COCO keypoint names in YOLO pose output order.
KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

# Bone pairs (index pairs into KEYPOINT_NAMES) drawn as the skeleton.
SKELETON_EDGES = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (0, 5),
    (0, 6),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at point b formed by points a-b-c, in degrees (0-180).

    Points are (x, y) coordinates. Returns NaN if either limb has zero length.
    """
    ba = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    bc = np.asarray(c, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    norm = np.linalg.norm(ba) * np.linalg.norm(bc)
    if norm == 0.0:
        return float("nan")
    cosine = np.clip(np.dot(ba, bc) / norm, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def normalize_keypoints(keypoints: np.ndarray) -> np.ndarray:
    """Normalize x, y keypoint coords to a torso-relative frame.

    Centres on the mid-hip point and scales by shoulder-to-hip distance so
    poses can be compared regardless of where the person stands in frame.
    Returns a copy; confidence column (if present) is preserved.
    """
    kpts = keypoints.astype(np.float64).copy()
    left_hip, right_hip = kpts[11, :2], kpts[12, :2]
    left_shoulder, right_shoulder = kpts[5, :2], kpts[6, :2]
    centre = (left_hip + right_hip) / 2.0
    scale = np.linalg.norm((left_shoulder + right_shoulder) / 2.0 - centre)
    if scale == 0.0:
        return kpts
    kpts[:, :2] = (kpts[:, :2] - centre) / scale
    return kpts
