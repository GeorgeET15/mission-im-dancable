"""Headless tests for the geometric dancer renderer."""

import math

import numpy as np
import pygame
import pytest

from mission_im_dancable.ui.dancer import build_dancer, draw_dancer, lerp_pose
from mission_im_dancable.ui.screens import PlayScreen


def _pose(conf=0.9, seed=0):
    k = np.zeros((17, 3))
    rng = np.random.default_rng(seed)
    k[:, :2] = rng.uniform(50, 250, (17, 2))
    k[:, 2] = conf
    # force a plausible body: shoulders/hips always present
    k[5] = (100, 50, conf); k[6] = (150, 50, conf)
    k[11] = (110, 150, conf); k[12] = (140, 150, conf)
    k[0] = (125, 20, conf)
    return k


def test_build_maps_all_landmarks_to_parts():
    parts = build_dancer(_pose(), 0.3)
    assert len(parts) >= 10  # torso facets + pelvis + head + limbs + studs
    assert all(len(p.points) >= 3 for p in parts)
    assert all(math.isfinite(v) for p in parts for pt in p.points for v in pt)


def test_missing_landmarks_skip_limbs_gracefully():
    k = _pose()
    for i in (7, 8, 9, 10, 13, 14, 15, 16):  # limbs gone; core survives
        k[i] = (0, 0, 0.0)
    parts = build_dancer(k, 0.3)
    assert len(parts) >= 3  # torso still drawn
    assert all(math.isfinite(v) for p in parts for pt in p.points for v in pt)


def test_low_confidence_skips_body_entirely():
    k = _pose(conf=0.1)
    assert build_dancer(k, 0.3) == []


def test_no_valid_pose_returns_empty():
    k = np.zeros((17, 3))
    assert build_dancer(k, 0.3) == []


def test_nan_coords_do_not_crash():
    k = _pose()
    k[5, 0] = np.nan
    parts = build_dancer(k, 0.3)
    assert all(math.isfinite(v) for p in parts for pt in p.points for v in pt)


def test_lerp_pose_midpoint_and_none_safety():
    a, b = _pose(seed=1), _pose(seed=2)
    mid = lerp_pose(a, b, 0.5)
    assert np.allclose(mid[:, :2], (a[:, :2] + b[:, :2]) / 2)
    assert lerp_pose(None, b, 0.5) is b
    assert lerp_pose(a, None, 0.5) is a


def test_draw_writes_pixels_and_respects_mirror():
    surf = pygame.Surface((200, 200))
    surf.fill((0, 0, 0))
    assert draw_dancer(pygame, surf, _pose(), (10, 10, 180, 180)) is True
    assert surf.get_bounding_rect().width > 0
    # mirrored render should produce different pixels
    surf2 = pygame.Surface((200, 200))
    draw_dancer(pygame, surf2, _pose(), (10, 10, 180, 180), mirror=True)
    assert not np.array_equal(pygame.surfarray.array3d(surf), pygame.surfarray.array3d(surf2))


def test_draw_scales_to_rect_and_never_mutates_input():
    k = _pose()
    before = k.copy()
    surf = pygame.Surface((400, 300))
    assert draw_dancer(pygame, surf, k, (0, 0, 400, 300)) is True
    assert np.array_equal(k, before)
    assert draw_dancer(pygame, surf, None, (0, 0, 400, 300)) is False
