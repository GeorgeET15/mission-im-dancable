import numpy as np
import pytest

from mission_im_dancable.beats import (
    BeatGrid,
    beats_cache_path,
    detect_beats,
    load_or_detect_beats,
)


def _grid(*beats):
    return BeatGrid(np.asarray(beats, dtype=np.float64))


def test_pulse_decays_after_beat():
    g = _grid(1.0, 2.0)
    assert g.pulse(1.0) == pytest.approx(1.0)
    assert 0.0 < g.pulse(1.5) < 1.0
    assert g.pulse(0.5) == 0.0  # before first beat


def test_phase_progress_between_beats():
    g = _grid(1.0, 2.0)
    assert g.phase(1.5) == pytest.approx(0.5)
    assert g.phase(2.0) <= 1.0
    assert g.phase(0.5) == pytest.approx(0.5)  # ramp toward first beat


def test_nearest_beat():
    g = _grid(1.0, 2.0, 3.0)
    assert g.nearest(1.7) == pytest.approx(2.0)
    assert g.nearest(0.1) == pytest.approx(1.0)


def test_empty_grid_is_safe():
    g = _grid()
    assert g.pulse(3.0) == 0.0
    assert g.phase(3.0) == 0.0
    assert g.nearest(3.0) is None


def test_detect_beats_on_synthetic_clicks():
    rate = 22050
    pcm = np.random.default_rng(0).normal(0, 0.01, rate * 6).astype(np.float32)
    for k in range(12):  # click every 0.5s (120 BPM)
        s = int(k * 0.5 * rate)
        pcm[s : s + 300] += 0.8
    beats = detect_beats(pcm, rate=rate)
    assert beats.size >= 8
    spacing = np.diff(beats).mean()
    assert 0.4 < spacing < 0.6


def test_detect_beats_handles_silence_and_none():
    assert detect_beats(None).size == 0
    assert detect_beats(np.zeros(22050, dtype=np.float32)).size == 0


def test_beat_cache_roundtrip_and_staleness(tmp_path):
    import json

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"v1")
    path = beats_cache_path(video, tmp_path / "cache")
    path.parent.mkdir(parents=True)
    meta = {"v": 1, "size": 2, "mtime": video.stat().st_mtime_ns}
    np.savez(path, meta=json.dumps(meta), beats=np.array([1.0, 2.0]))
    g = load_or_detect_beats(video, tmp_path / "cache")
    assert list(g.beats) == [1.0, 2.0]  # fresh cache used

    video.write_bytes(b"changed!")  # new size + mtime → stale → redetect
    g = load_or_detect_beats(video, tmp_path / "cache")
    assert g.beats.size == 0  # ffmpeg can't decode a fake video → empty grid
