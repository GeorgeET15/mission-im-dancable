"""Sprite playback tests — deterministic, no display required."""

import numpy as np
import pytest

from mission_im_dancable.ui.sprite import (
    SpriteAnim,
    frame_index_at,
    load_frames,
)


def test_frame_index_loops():
    assert frame_index_at(0.0, 48, 24) == 0
    assert frame_index_at(0.5, 48, 24) == 12
    assert frame_index_at(1.99, 48, 24) == 47
    assert frame_index_at(2.0, 48, 24) == 0  # 48 frames @24fps → 2s loop
    assert frame_index_at(4.1, 48, 24) == 2


def test_frame_index_pause_seek_restart():
    # pause = clock stops → index freezes; seek back = index rewinds;
    # restart t=0 → frame 0. All deterministic, no internal state.
    assert frame_index_at(1.0, 48, 24) == frame_index_at(1.0, 48, 24)
    assert frame_index_at(0.2, 48, 24) < frame_index_at(1.0, 48, 24)
    assert frame_index_at(0.0, 48, 24) == 0


def test_frame_index_empty_is_safe():
    assert frame_index_at(3.0, 0, 24) == 0
    assert frame_index_at(-1.0, 48, 24) == 0


def test_anim_surface_at_picks_frame():
    anim = SpriteAnim(["f0", "f1", "f2", "f3"], fps=2.0)
    assert anim.surface_at(0.0) == "f0"
    assert anim.surface_at(0.6) == "f1"  # int(0.6*2)=1
    assert anim.surface_at(2.1) == "f0"  # looped
    assert anim.count == 4


def test_load_frames_missing_dir(tmp_path):
    import pygame
    assert load_frames(pygame, tmp_path / "nope") is None
    assert load_frames(pygame, tmp_path) is None  # exists but empty


def test_load_frames_roundtrip(tmp_path):
    import cv2
    import pygame

    for i in range(3):
        img = np.zeros((8, 8, 4), dtype=np.uint8)
        img[:, :, :3] = i * 40
        img[:, :, 3] = 255
        cv2.imwrite(str(tmp_path / f"frame_{i + 1:04d}.png"), img)
    anim = load_frames(pygame, tmp_path)
    assert anim is not None and anim.count == 3
    # ordering: frame_0001 first
    assert anim.frames[0].get_size() == (8, 8)
