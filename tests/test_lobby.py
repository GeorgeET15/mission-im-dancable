"""Lobby music + UI sfx: synth shapes and safe wrappers."""

import numpy as np

from mission_im_dancable.lobby import LobbyMusic, lobby_samples
from mission_im_dancable.sfx import Sfx, click_samples
from mission_im_dancable.ui.widgets import Button


def test_lobby_samples_stereo_int16():
    arr = lobby_samples()
    assert arr.ndim == 2 and arr.shape[1] == 2
    assert arr.dtype == np.int16
    assert 8.0 < arr.shape[0] / 44100 < 12.0  # ~10 s loop
    assert np.abs(arr.astype(np.int32)).max() > 10000  # actually makes noise


def test_lobby_loop_ends_fade_out():
    arr = lobby_samples().astype(np.int32)    # abs of int16 min would wrap
    peak = np.abs(arr).max()
    # loop ends are faded well below the body so the wrap doesn't click
    assert np.abs(arr[:200]).max() < peak * 0.15
    assert np.abs(arr[-200:]).max() < peak * 0.15


def test_lobby_disabled_noops():
    m = LobbyMusic(enabled=False)
    m.ensure()
    m.stop()
    m.set_volume(0.9)
    assert m._channel is None


def test_click_samples_short_stereo():
    arr = click_samples()
    assert arr.ndim == 2 and arr.shape[1] == 2 and arr.dtype == np.int16
    assert arr.shape[0] / 44100 < 0.15  # a tap, not a tone


def test_sfx_disabled_noops():
    s = Sfx(enabled=False)
    s.click()
    s.set_volume(0.2)


def test_button_press_fires_hook():
    import pygame

    calls = []
    Button.on_press = lambda: calls.append(1)
    try:
        Button(pygame.Rect(0, 0, 10, 10), "x", "x").press()
        assert calls == [1]
    finally:
        Button.on_press = None


def test_button_press_survives_hook_errors():
    import pygame

    Button.on_press = lambda: 1 / 0
    try:
        Button(pygame.Rect(0, 0, 10, 10), "x", "x").press()  # must not raise
    finally:
        Button.on_press = None
