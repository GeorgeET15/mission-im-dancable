"""Low-poly 2.5D geometric dancer rendered with plain pygame polygons.

YOLO keypoints are 2D; the 'z' values below are conservative heuristics
for draw ordering and shading only — this is NOT 3D reconstruction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..geometry import normalize_keypoints
from ..scoring import mirror_keypoints

PALETTE = {
    "torso": (150, 80, 255),     # violet
    "pelvis": (110, 50, 210),
    "head": (255, 140, 200),     # pink
    "arm": (0, 200, 220),        # cyan
    "leg": (80, 130, 255),       # blue-violet
    "hand": (255, 60, 170),      # electric pink
    "foot": (0, 230, 180),       # teal
}
LIGHT = (0.45, -0.6)  # normalized-ish direction for facet shading


@dataclass
class Part:
    points: list[tuple[float, float]]
    color: tuple[int, int, int]
    z: float


def _shade(color, factor: float) -> tuple[int, int, int]:
    """factor>0 lightens, <0 darkens (simple directional lighting)."""
    return tuple(
        max(0, min(255, int(c + factor * 70))) for c in color
    )


def _norm(vx: float, vy: float) -> tuple[float, float]:
    d = math.hypot(vx, vy)
    return (vx / d, vy / d) if d > 1e-9 else (0.0, 1.0)


def _limb(a, b, w1: float, w2: float) -> list:
    """Tapered quad between points a→b with widths w1 (at a) and w2 (at b)."""
    dx, dy = _norm(b[0] - a[0], b[1] - a[1])
    nx, ny = -dy, dx
    return [
        (a[0] + nx * w1, a[1] + ny * w1),
        (a[0] - nx * w1, a[1] - ny * w1),
        (b[0] - nx * w2, b[1] - ny * w2),
        (b[0] + nx * w2, b[1] + ny * w2),
    ]


def _facet_factor(a, b) -> float:
    """Shade by how much the segment faces the light direction."""
    dx, dy = _norm(b[0] - a[0], b[1] - a[1])
    facing = -(dx * LIGHT[0] + dy * LIGHT[1])
    return max(-0.6, min(0.6, facing))


def _joint_z(kpts: np.ndarray, conf: float) -> float:
    """Tiny forward/back estimate: bent limbs come slightly forward."""
    return 0.0  # baseline; parts get z from their endpoints below


def lerp_pose(a: np.ndarray | None, b: np.ndarray | None, t: float) -> np.ndarray | None:
    """Interpolate between two (17,3) keypoint sets; None-safe."""
    if a is None:
        return b
    if b is None:
        return a
    out = b.copy()
    ok = (a[:, 2] > 0) & (b[:, 2] > 0)
    out[ok, :2] = a[ok, :2] * (1.0 - t) + b[ok, :2] * t
    return out


def build_dancer(keypoints: np.ndarray, conf_threshold: float) -> list[Part]:
    """Map a (17,3) normalized pose to a list of shaded Parts (back→front).

    Returns [] when too few landmarks exist to form a body — callers skip
    drawing rather than showing a broken avatar.
    """
    k = keypoints
    has = lambda i: k[i, 2] >= conf_threshold and np.isfinite(k[i, :2]).all()
    core = (5, 6, 11, 12)  # shoulders + hips must exist
    if sum(has(i) for i in core) < 3:
        return []

    P = lambda i: (float(k[i, 0]), float(k[i, 1]))
    parts: list[Part] = []

    # --- pelvis + torso ----------------------------------------------------
    if has(5) and has(6) and has(11) and has(12):
        pelvis = _limb(P(11), P(12), 0.28, 0.28)
        parts.append(Part(pelvis, _shade(PALETTE["pelvis"], 0.0), z=-0.1))
        chest = [P(5), P(6), P(12), P(11)]
        # two facets → faceted look
        parts.append(Part([chest[0], chest[1], chest[3]], _shade(PALETTE["torso"], 0.25), z=-0.05))
        parts.append(Part([chest[1], chest[2], chest[3]], _shade(PALETTE["torso"], -0.1), z=-0.05))

    # --- head (octagon, anchored by nose/ears or above shoulders) ----------
    if has(0):
        hx, hy = P(0)
        span = abs(k[5, 0] - k[6, 0]) if has(5) and has(6) else 0.4
        r = max(0.16 * span, 0.06)
        cy = hy - r * 0.6
        oct_pts = [
            (hx + r * math.cos(a), cy + r * math.sin(a))
            for a in [math.pi / 8 + i * math.pi / 4 for i in range(8)]
        ]
        parts.append(Part(oct_pts, _shade(PALETTE["head"], 0.1), z=0.05))
        # face facet highlight
        parts.append(Part(
            [(hx - r * 0.4, cy - r * 0.4), (hx + r * 0.4, cy - r * 0.4), (hx, cy + r * 0.15)],
            _shade(PALETTE["head"], 0.45), z=0.06,
        ))

    # --- limbs -------------------------------------------------------------
    limbs = [
        # (a, b, widths, color_key, z)
        (5, 7, 0.16, 0.13, "arm", 0.1),   # L upper arm
        (7, 9, 0.13, 0.10, "arm", 0.15),  # L forearm
        (6, 8, 0.16, 0.13, "arm", 0.1),   # R upper arm
        (8, 10, 0.13, 0.10, "arm", 0.15), # R forearm
        (11, 13, 0.20, 0.16, "leg", -0.15),  # L thigh
        (13, 15, 0.16, 0.12, "leg", -0.1),   # L shin
        (12, 14, 0.20, 0.16, "leg", -0.15),  # R thigh
        (14, 16, 0.16, 0.12, "leg", -0.1),   # R shin
    ]
    for a, b, w1, w2, color, z in limbs:
        if has(a) and has(b):
            shade = _facet_factor(P(a), P(b))
            parts.append(Part(_limb(P(a), P(b), w1, w2), _shade(PALETTE[color], shade), z))

    # --- hands / feet ------------------------------------------------------
    for idx, w, color, z in ((9, 0.09, "hand", 0.2), (10, 0.09, "hand", 0.2),
                             (15, 0.12, "foot", 0.0), (16, 0.12, "foot", 0.0)):
        if has(idx):
            x, y = P(idx)
            parts.append(Part(
                [(x - w, y - w), (x + w, y - w), (x + w, y + w), (x - w, y + w)],
                _shade(PALETTE[color], 0.1), z,
            ))

    # --- joint dots (elbows/knees read as poly studs) ----------------------
    for idx in (7, 8, 13, 14):
        if has(idx):
            x, y = P(idx)
            parts.append(Part(
                [(x, y - 0.07), (x + 0.07, y), (x, y + 0.07), (x - 0.07, y)],
                _shade((240, 240, 255), 0.0), z=0.25,
            ))

    parts.sort(key=lambda p: p.z)
    return parts


def draw_dancer(
    pygame,
    surf,
    keypoints: np.ndarray | None,
    rect: tuple[int, int, int, int],
    conf_threshold: float = 0.3,
    mirror: bool = False,
    torso_world: float = 2.2,
) -> bool:
    """Draw the dancer inside rect. Returns True if something was drawn.

    keypoints are normalized via normalize_keypoints first, so reference and
    live avatars render at comparable size regardless of camera distance.
    """
    if keypoints is None or keypoints[:, 2].max() <= 0:
        return False
    k = mirror_keypoints(keypoints, 0) if mirror else keypoints
    k = normalize_keypoints(k)
    x0, y0, w, h = rect
    scale = min(w, h) * 0.5 / torso_world  # torso_world units ≈ half panel
    cx, cy = x0 + w / 2, y0 + h * 0.62    # mid-hip anchor
    for part in build_dancer(k, conf_threshold):
        pts = [(cx + px * scale, cy + py * scale) for px, py in part.points]
        if all(math.isfinite(v) for p in pts for v in p):
            pygame.draw.polygon(surf, part.color, pts)
            pygame.draw.polygon(surf, (20, 12, 35), pts, 1)
    return True
