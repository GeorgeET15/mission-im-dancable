"""Beat extraction from video audio (ffmpeg + numpy) and beat-grid queries.

Decoding runs outside the render loop (background thread on the select
screen). Videos without audio produce an empty BeatGrid, never an error.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

BEAT_CACHE_VERSION = 1
SAMPLE_RATE = 22050


def _pcm(video_path: Path, rate: int = SAMPLE_RATE) -> np.ndarray | None:
    """Mono float32 PCM via ffmpeg; None when the video has no audio."""
    try:
        out = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(video_path), "-vn",
             "-ac", "1", "-ar", str(rate), "-f", "f32le", "-"],
            capture_output=True, timeout=180,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    pcm = np.frombuffer(out.stdout, dtype=np.float32)
    return pcm if pcm.size else None


def detect_beats(pcm: np.ndarray | None, rate: int = SAMPLE_RATE, hop: int = 512) -> np.ndarray:
    """Onset-energy beat detector → beat timestamps in seconds.

    Heuristic: RMS-energy onsets, tempo from autocorrelation over the
    60–180 BPM range, peaks picked per tempo-spaced window.
    """
    if pcm is None or pcm.size < rate:
        return np.array([])
    n = pcm.size // hop
    frames = pcm[: n * hop].reshape(n, hop)
    energy = np.sqrt((frames ** 2).mean(axis=1))
    onset = np.maximum(np.diff(energy, prepend=energy[0]), 0.0)
    if onset.max() <= 0:
        return np.array([])
    onset /= onset.max()
    fps = rate / hop
    centered = onset - onset.mean()
    lags = np.arange(max(1, int(fps * 60 / 180)), int(fps * 60 / 60) + 1)
    lags = lags[lags < n]
    tempo_lag = (
        lags[int(np.argmax([np.dot(centered[:-l], centered[l:]) for l in lags]))]
        if lags.size else int(fps)
    )
    step = max(int(tempo_lag * 0.5), 1)
    thresh = onset.mean() + 0.5 * onset.std()
    beats = [
        s + int(onset[s : s + step].argmax())
        for s in range(0, n, step)
        if onset[s : s + step].size and onset[s : s + step].max() >= thresh
    ]
    return np.asarray(beats, dtype=np.float64) / fps


@dataclass
class BeatGrid:
    """Query helpers around beat timestamps (seconds)."""

    beats: np.ndarray

    def pulse(self, t: float, decay: float = 4.0) -> float:
        """1.0 at a beat, exponentially decaying after it."""
        if self.beats.size == 0:
            return 0.0
        i = int(np.searchsorted(self.beats, t, side="right")) - 1
        if i < 0:
            return 0.0
        return float(np.exp(-decay * (t - self.beats[i])))

    def phase(self, t: float) -> float:
        """0→1 progress toward the next beat; 0 before the first beat."""
        if self.beats.size == 0:
            return 0.0
        i = int(np.searchsorted(self.beats, t, side="right")) - 1
        if i < 0:
            return min(t / self.beats[0], 1.0) if self.beats[0] > 0 else 0.0
        if i + 1 >= self.beats.size:
            return 1.0
        span = self.beats[i + 1] - self.beats[i]
        return min(max((t - self.beats[i]) / span, 0.0), 1.0) if span > 0 else 1.0

    def nearest(self, t: float) -> float | None:
        if self.beats.size == 0:
            return None
        return float(self.beats[int(np.argmin(np.abs(self.beats - t)))])


def _key(video_path: Path) -> str:
    return hashlib.sha1(str(video_path.resolve()).encode()).hexdigest()[:16]


def beats_cache_path(video_path: Path, cache_dir: Path) -> Path:
    return cache_dir / f"{video_path.stem}.{_key(video_path)}.beats.npz"


def load_or_detect_beats(video_path: Path, cache_dir: Path) -> BeatGrid:
    """Cached beats if fresh, else detect + cache. Empty grid on failure."""
    path = beats_cache_path(video_path, cache_dir)
    try:
        stat = video_path.stat()
    except OSError:
        return BeatGrid(np.array([]))
    if path.is_file():
        try:
            data = np.load(path, allow_pickle=False)
            meta = json.loads(str(data["meta"]))
            if (
                meta.get("v") == BEAT_CACHE_VERSION
                and meta.get("size") == stat.st_size
                and meta.get("mtime") == stat.st_mtime_ns
            ):
                return BeatGrid(np.asarray(data["beats"], dtype=np.float64))
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            pass  # stale/corrupt → redetect
    beats = detect_beats(_pcm(video_path))
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"v": BEAT_CACHE_VERSION, "size": stat.st_size, "mtime": stat.st_mtime_ns}
    np.savez(path, meta=json.dumps(meta), beats=beats)
    return BeatGrid(beats)
