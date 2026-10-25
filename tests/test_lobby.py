"""Lobby music: synth track shape + channel wrapper safety."""

import numpy as np

from mission_im_dancable.lobby import LobbyMusic, lobby_samples


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
