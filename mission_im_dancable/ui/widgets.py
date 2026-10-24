"""Reusable pygame widgets: buttons, toggles, bars, text field."""

from __future__ import annotations

import math

from . import theme


class Button:
    """Clickable button with hover, keyboard-focus, and press states."""

    def __init__(self, rect, label: str, action: str, primary: bool = False, enabled: bool = True):
        self.rect = rect
        self.label = label
        self.action = action
        self.primary = primary
        self.enabled = enabled
        self.hovered = False
        self.focused = False
        self._press_t = 0.0

    def contains(self, pos) -> bool:
        return self.rect.collidepoint(pos)

    def press(self) -> None:
        self._press_t = 0.15

    def update(self, dt: float) -> None:
        self._press_t = max(0.0, self._press_t - dt)

    def draw(self, pygame, surf, font, t: float) -> None:
        x, y, w, h = self.rect
        active = self.enabled and (self.hovered or self.focused)
        color = theme.PINK if self.primary else theme.CYAN
        if not self.enabled:
            color = theme.DIM
        scale = 0.96 if self._press_t > 0 else (1.0 + 0.03 * (1 + math.sin(t * 4)) if active and self.primary else 1.0)
        dx, dy = int(x + w * (1 - scale) / 2), int(y + h * (1 - scale) / 2)
        dw, dh = int(w * scale), int(h * scale)
        body = theme.PANEL if not self.primary else (60, 18, 70)
        pygame.draw.rect(surf, body, (dx, dy, dw, dh), border_radius=10)
        edge = 3 if active else 1
        pygame.draw.rect(surf, color, (dx, dy, dw, dh), edge, border_radius=10)
        if active:
            glow = pygame.Surface((dw + 16, dh + 16), pygame.SRCALPHA)
            pygame.draw.rect(glow, (*color, 60), glow.get_rect(), 4, border_radius=12)
            surf.blit(glow, (dx - 8, dy - 8))
        label = font.render(self.label, True, theme.TEXT if self.enabled else theme.DIM)
        surf.blit(label, label.get_rect(center=(x + w // 2, y + h // 2)))


class Toggle:
    """Labelled on/off option."""

    def __init__(self, rect, label: str, action: str, value: bool = False):
        self.rect = rect
        self.label = label
        self.action = action
        self.value = value
        self.hovered = self.focused = False

    def contains(self, pos) -> bool:
        return self.rect.collidepoint(pos)

    def draw(self, pygame, surf, font) -> None:
        x, y, w, h = self.rect
        txt = font.render(self.label, True, theme.TEXT)
        surf.blit(txt, (x, y + (h - txt.get_height()) // 2))
        box = (x + w - 60, y + h // 2 - 14, 60, 28)
        pygame.draw.rect(surf, theme.PANEL_EDGE, box, border_radius=14)
        knob_x = box[0] + (34 if self.value else 4)
        color = theme.CYAN if self.value else theme.DIM
        pygame.draw.circle(surf, color, (knob_x + 11, box[1] + 14), 12)


class ProgressBar:
    def __init__(self, rect, color=theme.CYAN, bg=theme.PANEL_EDGE):
        self.rect = rect
        self.color = color
        self.bg = bg
        self.shown = 0.0  # eased value

    def draw(self, pygame, surf, target: float, dt: float = 0.016) -> None:
        self.shown += (min(max(target, 0.0), 1.0) - self.shown) * min(dt * 8, 1.0)
        x, y, w, h = self.rect
        pygame.draw.rect(surf, self.bg, self.rect, border_radius=h // 2)
        if self.shown > 0.01:
            pygame.draw.rect(
                surf, self.color,
                (x, y, int(w * min(self.shown, 1.0)), h), border_radius=h // 2,
            )


class TextField:
    """Single-line editable text input (path entry fallback)."""

    def __init__(self, rect, text: str = "", placeholder: str = ""):
        self.rect = rect
        self.text = text
        self.placeholder = placeholder
        self.focused = False

    def contains(self, pos) -> bool:
        return self.rect.collidepoint(pos)

    def handle_key(self, key: int, unicode: str) -> bool:
        if not self.focused:
            return False
        if key == 8:  # backspace
            self.text = self.text[:-1]
            return True
        if unicode and unicode.isprintable() and key not in (13, 27, 9):
            self.text += unicode
            return True
        return False

    def draw(self, pygame, surf, font, t: float) -> None:
        pygame.draw.rect(surf, (16, 10, 30), self.rect, border_radius=8)
        edge = theme.PINK if self.focused else theme.PANEL_EDGE
        pygame.draw.rect(surf, edge, self.rect, 2, border_radius=8)
        shown = self.text[-46:] if self.text else self.placeholder
        color = theme.TEXT if self.text else theme.DIM
        txt = font.render(shown, True, color)
        surf.blit(txt, (self.rect.x + 10, self.rect.y + (self.rect.h - txt.get_height()) // 2))
        if self.focused and int(t * 2) % 2 == 0:
            cx = self.rect.x + 12 + txt.get_width()
            pygame.draw.line(surf, theme.CYAN, (cx, self.rect.y + 8), (cx, self.rect.y + self.rect.h - 8), 2)


class Stepper:
    """Numeric option with −/+ buttons: label on the left, controls right."""

    BTN = 46

    def __init__(self, rect, label: str, action: str, fmt=str):
        self.rect = rect
        self.label = label
        self.action = action  # "action-" / "action+" fired on the buttons
        self.fmt = fmt
        self.value = 0.0
        self.hovered = self.focused = False
        self._press_t = 0.0

    def _boxes(self):
        x, y, w, h = self.rect
        minus = (x + w - 2 * self.BTN - 90, y + (h - 36) // 2, self.BTN, 36)
        plus = (x + w - self.BTN, y + (h - 36) // 2, self.BTN, 36)
        val = (x + w - self.BTN - 86, y + (h - 36) // 2, 86, 36)
        return minus, val, plus

    def contains(self, pos) -> bool:
        return self.rect.collidepoint(pos)

    def action_at(self, pos) -> str | None:
        """Which control was clicked: 'action-', 'action+', or None."""
        minus, _, plus = self._boxes()
        import pygame
        if pygame.Rect(minus).collidepoint(pos):
            return f"{self.action}-"
        if pygame.Rect(plus).collidepoint(pos):
            return f"{self.action}+"
        return None

    def update(self, dt: float) -> None:
        self._press_t = max(0.0, self._press_t - dt)

    def draw(self, pygame, surf, font) -> None:
        x, y, w, h = self.rect
        txt = font.render(self.label, True, theme.TEXT if not self.focused else theme.CYAN)
        surf.blit(txt, (x, y + (h - txt.get_height()) // 2))
        minus, val, plus = self._boxes()
        for box, glyph in ((minus, "−"), (plus, "+")):
            pygame.draw.rect(surf, theme.PANEL, box, border_radius=8)
            pygame.draw.rect(surf, theme.PANEL_EDGE, box, 1, border_radius=8)
            g = font.render(glyph, True, theme.CYAN)
            surf.blit(g, g.get_rect(center=(box[0] + box[2] // 2, box[1] + box[3] // 2)))
        vtxt = font.render(self.fmt(self.value), True, theme.GOLD)
        surf.blit(vtxt, vtxt.get_rect(center=(val[0] + val[2] // 2, val[1] + val[3] // 2)))


class TimelineBar:
    """Seekable dance timeline with per-segment blocks.

    Segments are colored by state: done (already played), selected (the
    move being studied), active (under the playhead) and upcoming. The
    screen maps x positions to seconds via fraction_at(); it never seeks
    on its own.
    """

    def __init__(self, rect):
        self.rect = rect

    def contains(self, pos) -> bool:
        return self.rect.collidepoint(pos)

    def fraction_at(self, pos) -> float | None:
        """0-1 position along the bar for a click, or None outside."""
        if not self.rect.collidepoint(pos):
            return None
        return min(max((pos[0] - self.rect.x) / self.rect.w, 0.0), 1.0)

    def draw(
        self,
        pygame,
        surf,
        font,
        time_s: float,
        duration: float,
        segments=None,
        state_of=None,
        interactive: bool = True,
    ) -> None:
        """Draw track + segment blocks + playhead.

        segments: list of objects with .start/.end/.name (Move works).
        state_of(i) -> "done" | "selected" | "active" | "upcoming";
        segments without a callback all draw as plain blocks.
        """
        x, y, w, h = self.rect
        duration = max(duration, 1e-6)
        pygame.draw.rect(surf, theme.PANEL, self.rect, border_radius=h // 2)
        if segments:
            for i, seg in enumerate(segments):
                sx = x + int(w * seg.start / duration)
                sw = max(int(w * (seg.end - seg.start) / duration) - 2, 2)
                block = pygame.Rect(sx, y + 2, sw, h - 4)
                state = state_of(i) if state_of else "upcoming"
                if state == "done":
                    fill = (90, 60, 140)
                    edge = theme.PANEL_EDGE
                elif state == "selected":
                    fill = (30, 90, 110)
                    edge = theme.CYAN
                elif state == "active":
                    fill = (70, 40, 100)
                    edge = theme.GOLD
                else:
                    fill = (40, 24, 66)
                    edge = theme.PANEL_EDGE
                pygame.draw.rect(surf, fill, block, border_radius=4)
                pygame.draw.rect(surf, edge, block, 1 if state == "upcoming" else 2,
                                 border_radius=4)
        else:
            fill_w = int(w * min(time_s / duration, 1.0))
            if fill_w:
                pygame.draw.rect(surf, (70, 40, 100), (x, y + 2, fill_w, h - 4),
                                 border_radius=h // 2)
        pygame.draw.rect(surf, theme.PANEL_EDGE, self.rect, 1, border_radius=h // 2)
        # playhead
        px = x + int(w * min(max(time_s / duration, 0.0), 1.0))
        pygame.draw.line(surf, theme.TEXT, (px, y - 4), (px, y + h + 4), 2)
        pygame.draw.polygon(
            surf, theme.TEXT,
            [(px - 5, y - 4), (px + 5, y - 4), (px, y + 2)],
        )
        if interactive:
            label = font.render(f"{time_s:5.1f}s / {duration:5.1f}s", True, theme.DIM)
            surf.blit(label, label.get_rect(midright=(x + w, y - 10)))


class FocusGroup:
    """Keyboard focus navigation over a list of focusable widgets."""

    def __init__(self, items):
        self.items = [i for i in items if i is not None]
        self.index = 0
        self._sync()

    def _sync(self):
        for i, item in enumerate(self.items):
            item.focused = i == self.index

    def move(self, delta: int) -> None:
        if not self.items:
            return
        self.index = (self.index + delta) % len(self.items)
        self._sync()

    @property
    def current(self):
        return self.items[self.index] if self.items else None

    def hover(self, pos) -> None:
        for i, item in enumerate(self.items):
            item.hovered = item.contains(pos)
            if item.hovered:
                self.index = i
        self._sync()
