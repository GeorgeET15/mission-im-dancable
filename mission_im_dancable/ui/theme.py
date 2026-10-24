"""Visual theme: neon dance-arcade palette and animated background pieces."""

from __future__ import annotations

import math
import random

# Palette — deep purple / near-black with electric pink + cyan accents.
BG_TOP = (24, 10, 42)
BG_BOT = (8, 4, 18)
PANEL = (28, 16, 48)
PANEL_EDGE = (80, 50, 130)
PINK = (255, 60, 170)
CYAN = (0, 230, 220)
VIOLET = (150, 80, 255)
GOLD = (255, 205, 90)
TEXT = (240, 236, 250)
DIM = (150, 140, 175)
GOOD = (80, 255, 160)
MISS = (255, 80, 95)

JUDGE_COLORS = {"PERFECT": GOLD, "GREAT": CYAN, "MISS": MISS}


def gradient_surface(pygame, width: int, height: int, top=BG_TOP, bottom=BG_BOT):
    """Vertical gradient background surface (cache it — same every frame)."""
    surf = pygame.Surface((1, height))
    for y in range(height):
        f = y / max(height - 1, 1)
        surf.set_at(
            (0, y),
            tuple(int(top[i] + (bottom[i] - top[i]) * f) for i in range(3)),
        )
    return pygame.transform.scale(surf, (width, height))


def draw_dance_floor(pygame, surf, t: float, horizon: int, color=VIOLET) -> None:
    """Perspective grid floor — a stylized neon dance floor."""
    w, h = surf.get_size()
    for i in range(-6, 7):  # radiating lines
        x = w // 2 + i * (w // 6)
        pygame.draw.line(surf, (*color[:3],), (w // 2, horizon), (x, h), 1)
    for j in range(1, 6):  # moving horizontal rows
        p = ((t * 0.35 + j / 6) % 1.0) ** 2.2
        y = int(horizon + (h - horizon) * p)
        if y < h:
            alpha = int(60 + 140 * p)
            row = pygame.Surface((w, 2), pygame.SRCALPHA)
            row.fill((*color, alpha))
            surf.blit(row, (0, y))


def draw_spotlights(pygame, surf, t: float, y_top: int = 0) -> None:
    """Two rotating translucent light cones from the top corners."""
    w, h = surf.get_size()
    for i, (x0, phase, color) in enumerate(
        ((w * 0.2, 0.0, PINK), (w * 0.8, math.pi, CYAN))
    ):
        ang = math.sin(t * 0.7 + phase) * 0.5
        tip = (x0, y_top)
        far = h * 1.1
        dx = math.sin(ang) * far
        cone = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.polygon(
            cone,
            (*color, 26),
            [tip, (x0 + dx - w * 0.16, far), (x0 + dx + w * 0.16, far)],
        )
        surf.blit(cone, (0, 0))


def draw_equalizer(pygame, surf, t: float, rect, bars: int = 40, color=CYAN) -> None:
    """Animated equalizer bars along a rect — pure eye candy."""
    x, y, w, h = rect
    bw = w / bars
    for i in range(bars):
        amp = 0.25 + 0.75 * abs(
            math.sin(t * 2.4 + i * 0.55) * math.sin(t * 0.9 + i * 1.3)
        )
        bh = int(h * amp)
        pygame.draw.rect(
            surf, color, (x + i * bw + 1, y + h - bh, max(int(bw) - 2, 1), bh)
        )


def glow_text(pygame, font, s: str, color, glow=VIOLET, passes: int = 2):
    """Render text with a soft neon glow behind it."""
    base = font.render(s, True, color)
    w, h = base.get_size()
    out = pygame.Surface((w + passes * 8, h + passes * 8), pygame.SRCALPHA)
    halo = font.render(s, True, (*glow, 90))
    for r in range(passes, 0, -1):
        for dx, dy in ((-r * 2, 0), (r * 2, 0), (0, -r * 2), (0, r * 2)):
            out.blit(halo, (r * 4 + dx, r * 4 + dy))
    out.blit(base, (passes * 4, passes * 4))
    return out


class Particles:
    """Tiny celebratory particle bursts."""

    def __init__(self) -> None:
        self._bits: list[list] = []

    def burst(self, x: float, y: float, color, n: int = 26, speed: float = 260.0) -> None:
        for _ in range(n):
            ang = random.uniform(0, math.tau)
            v = random.uniform(0.3, 1.0) * speed
            self._bits.append(
                [x, y, math.cos(ang) * v, math.sin(ang) * v - 60,
                 random.uniform(0.5, 1.0), color]
            )

    def update(self, dt: float) -> None:
        for b in self._bits:
            b[0] += b[2] * dt
            b[1] += b[3] * dt
            b[3] += 300 * dt  # gravity
            b[4] -= dt
        self._bits = [b for b in self._bits if b[4] > 0]

    def draw(self, pygame, surf) -> None:
        for x, y, _, _, life, color in self._bits:
            r = max(int(4 * life), 1)
            pygame.draw.circle(surf, color, (int(x), int(y)), r)

    def clear(self) -> None:
        self._bits.clear()
