"""Dance-arcade app entry: window, event loop, screen navigation."""

from __future__ import annotations

import time

from ..config import AppConfig
from . import theme
from .nav import Navigator
from .screens import AppContext, HomeScreen, W, H


def _fonts(pygame):
    def f(size, bold=False):
        return pygame.font.SysFont("ubuntumono,dejavusansmono,liberationmono", size, bold=bold)

    return {"sm": f(17), "med": f(22, bold=True), "big": f(48, bold=True), "title": f(46, bold=True)}


def run_arcade(cfg: AppConfig) -> int:
    """Launch the dance-arcade UI. Returns a process exit code."""
    import pygame

    pygame.init()
    pygame.display.set_caption("MISSION IM-DANCABLE")
    window = pygame.display.set_mode((W, H), pygame.RESIZABLE)
    canvas = pygame.Surface((W, H))
    clock = pygame.time.Clock()

    manager = Navigator()
    ctx = AppContext(cfg, _fonts(pygame), manager)
    ctx.bg = theme.gradient_surface(pygame, W, H)
    manager.replace(HomeScreen())

    running = True
    while running and not ctx.quit:
        dt = clock.tick(60) / 1000.0
        ctx.t = time.perf_counter() % 1000.0  # wrap-safe animation clock
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
                break
            if ev.type in (pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN):
                # translate window coords into canvas coords (letterbox scale)
                ww, wh = window.get_size()
                s = min(ww / W, wh / H)
                if s <= 0:
                    continue
                pos = ev.pos
                ev = pygame.event.Event(
                    ev.type,
                    pos=((pos[0] - (ww - W * s) / 2) / s, (pos[1] - (wh - H * s) / 2) / s),
                    **({"button": ev.button} if ev.type == pygame.MOUSEBUTTONDOWN else {}),
                )
            manager.handle_event(ctx, ev)
        manager.update(ctx, dt)
        ctx.particles.update(dt)
        manager.draw(ctx, canvas)
        # scale canvas onto the actual window (handles resize gracefully)
        ww, wh = window.get_size()
        s = min(ww / W, wh / H)
        scaled = pygame.transform.scale(canvas, (int(W * s), int(H * s)))
        window.fill(theme.BG_BOT)
        window.blit(scaled, ((ww - W * s) / 2, (wh - H * s) / 2))
        pygame.display.flip()
    pygame.quit()
    return 0
