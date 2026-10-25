"""Reference-video audio: extraction to a cached WAV + segment playback.

The game plays the dance video's real audio. ffmpeg decodes it once into
cache/ as mono 44.1 kHz WAV; pygame.mixer.music plays a sub-range of it
via play(start=...) so each move's soundtrack matches the skeleton.
Videos without an audio track produce no WAV and no error.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

AUDIO_CACHE_VERSION = 1
AUDIO_RATE = 44100


def _key(video_path: Path) -> str:
    return hashlib.sha1(str(video_path.resolve()).encode()).hexdigest()[:16]


def audio_cache_path(video_path: Path, cache_dir: Path) -> Path:
    return cache_dir / f"{video_path.stem}.{_key(video_path)}.wav"


def _meta_path(wav_path: Path) -> Path:
    return wav_path.with_suffix(wav_path.suffix + ".json")


def _fresh(wav_path: Path, video_path: Path) -> bool:
    try:
        meta = json.loads(_meta_path(wav_path).read_text())
        stat = video_path.stat()
        return (
            meta.get("v") == AUDIO_CACHE_VERSION
            and meta.get("size") == stat.st_size
            and meta.get("mtime") == stat.st_mtime_ns
        )
    except (OSError, json.JSONDecodeError):
        return False


def ensure_audio_wav(video_path: Path, cache_dir: Path) -> Path | None:
    """Decode the video's audio track to a cached WAV; None if none/fails.

    Call from a worker thread — decoding blocks for a moment.
    """
    try:
        stat = video_path.stat()
    except OSError:
        return None
    wav = audio_cache_path(video_path, cache_dir)
    if wav.is_file() and _fresh(wav, video_path):
        return wav
    try:
        wav.parent.mkdir(parents=True, exist_ok=True)
        out = subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(video_path),
             "-vn", "-ac", "1", "-ar", str(AUDIO_RATE), str(wav)],
            capture_output=True, timeout=180,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0 or not wav.is_file() or wav.stat().st_size < 1000:
        try:
            wav.unlink(missing_ok=True)
        except OSError:
            pass
        return None  # no audio track (or broken decode)
    _meta_path(wav).write_text(json.dumps(
        {"v": AUDIO_CACHE_VERSION, "size": stat.st_size, "mtime": stat.st_mtime_ns}
    ))
    return wav


class Music:
    """Thin pygame.mixer.music wrapper used by gameplay screens.

    All methods are safe when audio is unavailable (no device, no WAV):
    enabled stays False and every call is a no-op.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._loaded: Path | None = None
        self._base = 0.0      # audio-file time where playback started
        self._volume = 0.8

    # -- internal -----------------------------------------------------------

    def _music(self):
        import pygame
        return pygame.mixer.music

    # -- control ------------------------------------------------------------

    def set_volume(self, fraction: float) -> None:
        self._volume = min(max(float(fraction), 0.0), 1.0)
        if self.enabled:
            self._music().set_volume(self._volume)

    @property
    def loaded(self) -> bool:
        return self._loaded is not None

    @property
    def path(self) -> Path | None:
        """Which WAV is currently loaded — screens compare against
        ctx.audio_path so switching videos re-loads the right track."""
        return self._loaded

    def load(self, wav_path: Path | None) -> bool:
        """Load a WAV into the music channel.

        Idempotent per path: loading the track that's already loaded is a
        cheap no-op; a *different* path stops and replaces the old track.
        """
        if not self.enabled or wav_path is None:
            return False
        wav_path = Path(wav_path)
        if self._loaded == wav_path:
            return True
        try:
            self._music().stop()
            self._music().load(str(wav_path))
            self._music().set_volume(self._volume)
            self._loaded = wav_path
            return True
        except Exception:
            self._loaded = None
            return False

    def play_at(self, t: float) -> None:
        """Start (or re-start) playback at t seconds into the loaded track."""
        if not self.enabled or self._loaded is None:
            return
        try:
            self._music().play(start=max(float(t), 0.0))
            self._base = float(t)
        except Exception:
            pass

    def pause(self) -> None:
        if self.enabled:
            self._music().pause()

    def unpause(self) -> None:
        if self.enabled:
            self._music().unpause()

    def stop(self) -> None:
        if self.enabled:
            self._music().stop()

    def position(self) -> float | None:
        """Current position inside the audio file (seconds), or None."""
        if not self.enabled or self._loaded is None:
            return None
        pos = self._music().get_pos()
        if pos < 0:
            return None
        return self._base + pos / 1000.0
