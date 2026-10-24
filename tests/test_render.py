import numpy as np

from mission_im_dancable.geometry import SKELETON_EDGES
from mission_im_dancable.render import draw_hud, draw_pose


def _pose() -> np.ndarray:
    kpts = np.zeros((17, 3))
    for i in range(17):
        kpts[i] = [40 + i * 10, 60 + i * 5, 0.9]
    return kpts


def test_draw_pose_paints_skeleton():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    draw_pose(frame, _pose(), conf_threshold=0.5)
    assert frame.sum() > 0


def test_draw_pose_skips_low_confidence():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    kpts = _pose()
    kpts[:, 2] = 0.1
    draw_pose(frame, kpts, conf_threshold=0.5)
    assert frame.sum() == 0


def test_draw_hud_paints_text():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    draw_hud(frame, fps=30.0, inference_ms=12.3, device="cuda", person_detected=True)
    assert frame.sum() > 0


def test_draw_hud_no_detection():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    draw_hud(frame, fps=30.0, inference_ms=None, device="cpu", person_detected=False)
    assert frame.sum() > 0


def test_skeleton_edges_valid():
    assert len(SKELETON_EDGES) == 18
