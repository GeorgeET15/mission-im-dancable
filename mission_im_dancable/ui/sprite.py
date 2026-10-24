"""Amy sprite playback: pre-rendered character frames synced to a clock.

Frames are rendered from Blender (assets/blender/frames/frame_NNNN.png at
SPRITE_FPS). The component is deterministic: the displayed frame derives
from the caller's clock time, so pause/seek/restart work by driving the
shared session time — the sprite never keeps its own clock.
"""

from __future__ import annotations

from pathlib import Path

SPRITE_DIR = Path(__file__).resolve().parent.parent.parent / "assets" / "blender" / "frames"
SPRITE_FPS = 24.0  # matches the .blend frame rate


def frame_index_at(t: float, n_frames: int, fps: float = SPRITE_FPS) -> int:
    """Looping frame index for time t; 0 for empty clips or t<=0."""
    if n_frames <= 0 or t <= 0.0:
        return 0
    return int(t * fps) % n_frames


class SpriteAnim:
    """Loaded frame surfaces + clock-synced drawing."""

    def __init__(self, frames: list, fps: float = SPRITE_FPS) -> None:
        self.frames = frames
        self.fps = fps

    @property
    def count(self) -> int:
        return len(self.frames)

    def index_at(self, t: float) -> int:
        return frame_index_at(t, self.count, self.fps)

    def surface_at(self, t: float):
        return self.frames[self.index_at(t)] if self.frames else None

    def draw(self, pygame, surf, rect, t: float) -> None:
        """Scale (aspect-preserving, fit height) and center into rect."""
        frame = self.surface_at(t)
        if frame is None:
            return
        x, y, w, h = rect
        fw, fh = frame.get_size()
        scale = min(w / fw, h / fh)
        dw, dh = int(fw * scale), int(fh * scale)
        scaled = pygame.transform.smoothscale(frame, (dw, dh))
        surf.blit(scaled, (x + (w - dw) // 2, y + (h - dh) // 2))


def load_frames(pygame, directory: Path = SPRITE_DIR, fps: float = SPRITE_FPS) -> SpriteAnim | None:
    """Load sorted frame_NNNN.png files; None when missing/empty."""
    try:
        paths = sorted(directory.glob("frame_*.png"))
    except OSError:
        return None
    if not paths:
        return None
    try:
        frames = [pygame.image.load(str(p)) for p in paths]
    except Exception:
        return None
    return SpriteAnim(frames, fps)
