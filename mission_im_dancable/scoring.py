"""Confidence-aware pose comparison for practice mode.

Poses are normalized (hip-centred, torso-scaled) before comparison so the
score is invariant to where the dancer stands and how big they appear.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import KEYPOINT_NAMES, joint_angle, normalize_keypoints

# Left/right keypoint index pairs swapped when mirroring the reference.
MIRROR_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]

# Joints compared by angle: (name, a, b, c) measures the angle at b.
JOINTS = [
    ("left elbow", 5, 7, 9),
    ("right elbow", 6, 8, 10),
    ("left shoulder", 11, 5, 7),
    ("right shoulder", 12, 6, 8),
    ("left hip", 5, 11, 13),
    ("right hip", 6, 12, 14),
    ("left knee", 11, 13, 15),
    ("right knee", 12, 14, 16),
]

ANGLE_MISMATCH_DEG = 25.0  # angle error above this counts as a real mismatch

SCORE_SCALE = 4.0  # exp(-SCORE_SCALE * mean_error) → higher = stricter
MIN_RELIABLE_KEYPOINTS = 6  # fewer than this and we refuse to score
MISMATCH_MIN_ERROR = 0.15  # normalized error below this = "close enough"


def mirror_keypoints(keypoints: np.ndarray, axis_width: float) -> np.ndarray:
    """Return a left-right mirrored copy of (N, 3+) keypoints.

    Swaps left/right body indices and reflects x around axis_width/2.
    Pass axis_width=0 for hip-centred normalized coordinates (x → -x).
    """
    out = keypoints.copy()
    for left, right in MIRROR_PAIRS:
        out[[left, right]] = out[[right, left]]
    out[:, 0] = axis_width - out[:, 0]
    return out


@dataclass
class PoseScore:
    """Result of comparing a live pose to one reference pose.

    keypoint_errors holds per-keypoint normalized distances; entries for
    unreliable landmarks are NaN so they can never win "worst mismatch".
    """

    score: float
    keypoint_errors: np.ndarray
    reliable_count: int


def score_pose(
    live: np.ndarray,
    reference: np.ndarray,
    conf_threshold: float,
    min_reliable: int = MIN_RELIABLE_KEYPOINTS,
) -> PoseScore | None:
    """Score live keypoints against reference keypoints in [0, 1].

    Only landmarks where BOTH poses have confidence >= conf_threshold are
    compared; low-confidence points are ignored rather than trusted.
    Returns None when too few reliable landmarks exist to judge.
    """
    reliable = (live[:, 2] >= conf_threshold) & (reference[:, 2] >= conf_threshold)
    reliable_count = int(reliable.sum())
    if reliable_count < min_reliable:
        return None

    live_n = normalize_keypoints(live)
    ref_n = normalize_keypoints(reference)
    errors = np.full(live.shape[0], np.nan)
    errors[reliable] = np.linalg.norm(
        live_n[reliable, :2] - ref_n[reliable, :2], axis=1
    )
    score = float(np.exp(-SCORE_SCALE * float(np.nanmean(errors))))
    return PoseScore(score=score, keypoint_errors=errors, reliable_count=reliable_count)


def best_window_score(
    live: np.ndarray,
    reference_window: np.ndarray,
    conf_threshold: float,
) -> tuple[PoseScore, int] | None:
    """Score the live pose against each candidate reference pose in a window.

    reference_window has shape (M, NUM_KEYPOINTS, 3). Rows with no detection
    (all-zero confidence) are skipped. Returns (best PoseScore, row index).
    """
    best: tuple[PoseScore, int] | None = None
    for i in range(reference_window.shape[0]):
        ref = reference_window[i]
        if ref[:, 2].max() <= 0:
            continue
        result = score_pose(live, ref, conf_threshold)
        if result is not None and (best is None or result.score > best[0].score):
            best = (result, i)
    return best


# COCO indices 0–4 are face landmarks (nose, eyes, ears). They are excluded
# from correction hints — "move your eye up-left" is not actionable feedback.
FACE_KEYPOINTS = 5


def worst_mismatch(pose_score: PoseScore, min_error: float = MISMATCH_MIN_ERROR) -> int | None:
    """Keypoint index of the largest reliable limb error, or None if all close.

    Face landmarks are ignored — only shoulders, elbows, wrists, hips, knees
    and ankles can win a correction hint.
    """
    errors = pose_score.keypoint_errors.copy()
    errors[:FACE_KEYPOINTS] = np.nan
    if not np.isfinite(errors).any():
        return None
    idx = int(np.nanargmax(errors))
    return idx if errors[idx] >= min_error else None


def describe_correction(
    keypoint_index: int,
    live: np.ndarray,
    reference: np.ndarray,
) -> str:
    """Human-readable correction, e.g. 'left wrist: move up-left'.

    Directions describe where the landmark should move in the displayed
    (selfie-view) image, so 'left' means toward screen left.
    """
    live_n = normalize_keypoints(live)
    ref_n = normalize_keypoints(reference)
    dx, dy = ref_n[keypoint_index, :2] - live_n[keypoint_index, :2]
    parts: list[str] = []
    if dy < -0.05:
        parts.append("up")
    elif dy > 0.05:
        parts.append("down")
    if dx < -0.05:
        parts.append("left")
    elif dx > 0.05:
        parts.append("right")
    direction = "-".join(parts) if parts else "adjust"
    return f"{KEYPOINT_NAMES[keypoint_index].replace('_', ' ')}: move {direction}"


def _joint_conf_ok(kpts: np.ndarray, indices: tuple[int, int, int], conf_threshold: float) -> bool:
    return all(kpts[i, 2] >= conf_threshold for i in indices)


def angle_errors(
    live: np.ndarray, reference: np.ndarray, conf_threshold: float
) -> dict[str, float]:
    """Absolute joint-angle differences in degrees for reliable joints.

    A joint counts only when all three landmarks clear the confidence
    threshold in BOTH poses. Unreliable or degenerate joints are omitted.
    """
    errors: dict[str, float] = {}
    for name, a, b, c in JOINTS:
        if not (
            _joint_conf_ok(live, (a, b, c), conf_threshold)
            and _joint_conf_ok(reference, (a, b, c), conf_threshold)
        ):
            continue
        live_deg = joint_angle(live[a, :2], live[b, :2], live[c, :2])
        ref_deg = joint_angle(reference[a, :2], reference[b, :2], reference[c, :2])
        if np.isfinite(live_deg) and np.isfinite(ref_deg):
            errors[name] = abs(live_deg - ref_deg)
    return errors


@dataclass
class PoseAnalysis:
    """Combined positional + joint-angle comparison of two poses."""

    pose_score: PoseScore
    angle_errors: dict[str, float]

    def worst_angle(self) -> tuple[str, float] | None:
        """(joint name, error degrees) of the worst joint, or None."""
        if not self.angle_errors:
            return None
        name = max(self.angle_errors, key=self.angle_errors.get)
        if self.angle_errors[name] < ANGLE_MISMATCH_DEG:
            return None
        return name, self.angle_errors[name]


def analyze_pose(
    live: np.ndarray, reference: np.ndarray, conf_threshold: float
) -> PoseAnalysis | None:
    """Score a pose positionally and by joint angles. None when unreliable."""
    pose_score = score_pose(live, reference, conf_threshold)
    if pose_score is None:
        return None
    return PoseAnalysis(
        pose_score=pose_score,
        angle_errors=angle_errors(live, reference, conf_threshold),
    )


def describe_angle_correction(
    joint_name: str, live_deg: float, ref_deg: float
) -> str:
    """Correction hint for a joint angle, e.g. 'left elbow: straighten'."""
    if live_deg < ref_deg:
        return f"{joint_name}: straighten more"
    return f"{joint_name}: bend more"


def joint_angle_deg(
    kpts: np.ndarray, name: str
) -> float | None:
    """Live angle in degrees for a named joint, or None if unavailable."""
    for jname, a, b, c in JOINTS:
        if jname == name:
            deg = joint_angle(kpts[a, :2], kpts[b, :2], kpts[c, :2])
            return deg if np.isfinite(deg) else None
    return None


class ScoreSmoother:
    """Exponential moving average so the displayed score does not jitter."""

    def __init__(self, alpha: float = 0.25) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self._alpha = alpha
        self._value: float | None = None

    def update(self, sample: float | None) -> float | None:
        """Fold in a new sample; None keeps the previous smoothed value."""
        if sample is None:
            return self._value
        if self._value is None:
            self._value = sample
        else:
            self._value = self._alpha * sample + (1.0 - self._alpha) * self._value
        return self._value

    @property
    def value(self) -> float | None:
        return self._value


@dataclass
class CorrectionTracker:
    """Holds one correction until a different mismatch persists.

    A new candidate must win `hold_frames` consecutive updates before the
    displayed correction switches, so the hint does not flicker every frame.
    """

    hold_frames: int = 10
    _current: int | str | None = None
    _candidate: int | str | None = None
    _candidate_count: int = 0

    def update(self, mismatch: int | str | None) -> int | str | None:
        # No current correction → adopt the first mismatch immediately;
        # hysteresis only protects against *changing* the hint every frame.
        if self._current is None and mismatch is not None:
            self._current = mismatch
            self._candidate = None
            self._candidate_count = 0
            return self._current
        if mismatch == self._current:
            self._candidate = None
            self._candidate_count = 0
            return self._current
        if mismatch == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate = mismatch
            self._candidate_count = 1
        if self._candidate_count >= self.hold_frames:
            self._current = self._candidate
            self._candidate = None
            self._candidate_count = 0
        return self._current

    @property
    def current(self) -> int | str | None:
        return self._current
