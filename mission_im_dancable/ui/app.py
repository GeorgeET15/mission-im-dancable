"""Dance-arcade app entry: window, event loop, screen navigation."""

from __future__ import annotations

import time

from ..audio import Music
from ..config import AppConfig
from ..persist import save_settings
from . import theme
from .nav import Navigator
from .screens import (
    AppContext,
    BrowseScreen,
    HomeScreen,
    ScoresScreen,
    SelectScreen,
    SettingsScreen,
    W,
    H,
    close_camera,
)
from ..lobby import LobbyMusic

_MENU_SCREENS = (HomeScreen, SelectScreen, BrowseScreen, SettingsScreen, ScoresScreen)


def _fonts(pygame):
    def f(size, bold=False):
        return pygame.font.SysFont("ubuntumono,dejavusansmono,liberationmono", size, bold=bold)

    def g(size, bold=False):  # title uses a cleaner sans — mono glyphs were illegible
        return pygame.font.SysFont("ubuntusans,dejavusans,liberationsans,arial", size, bold=bold)

    return {"sm": f(18), "med": f(24, bold=True), "big": f(56, bold=True), "title": g(64, bold=True)}


def _apply_fullscreen(pygame, on: bool):
    flags = pygame.FULLSCREEN if on else pygame.RESIZABLE
    size = (0, 0) if on else (W, H)
    return pygame.display.set_mode(size, flags)


def run_arcade(cfg: AppConfig) -> int:
    """Launch the dance-arcade UI. Returns a process exit code."""
    import pygame

    pygame.init()
    try:
        pygame.mixer.init()
        audio_ok = True
    except Exception:
        audio_ok = False  # no audio device — game still runs silent
    pygame.display.set_caption("MISSION IM-DANCABLE")
    window = pygame.display.set_mode((W, H), pygame.RESIZABLE)
    canvas = pygame.Surface((W, H))
    clock = pygame.time.Clock()

    manager = Navigator()
    ctx = AppContext(cfg, _fonts(pygame), manager)
    ctx.bg = theme.gradient_surface(pygame, W, H)
    ctx.music = Music(enabled=audio_ok)
    ctx.music.set_volume(float(ctx.settings.get("volume", 80)) / 100.0)
    ctx.lobby = LobbyMusic(enabled=audio_ok,
                           volume=float(ctx.settings.get("volume", 80)) / 250.0)
    applied_fullscreen = False
    if ctx.fullscreen:
        window = _apply_fullscreen(pygame, True)
        applied_fullscreen = True
    manager.replace(HomeScreen())

    running = True
    try:
        while running and not ctx.quit:
            dt = clock.tick(60) / 1000.0
            ctx.t = time.perf_counter() % 1000.0  # wrap-safe animation clock
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                    break
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_F11:
                    ctx.fullscreen = not ctx.fullscreen
                    ctx.settings["fullscreen"] = ctx.fullscreen
                    save_settings(ctx.settings)
                    continue
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
            if manager.current is None:
                running = False  # stack underflow — nothing left to show
                break
            # lobby loop lives only on menus — dance screens own the audio
            if isinstance(manager.current, _MENU_SCREENS):
                ctx.lobby.ensure()
            else:
                ctx.lobby.stop()
            ctx.particles.update(dt)
            manager.draw(ctx, canvas)
            # scale canvas onto the actual window (handles resize gracefully)
            ww, wh = window.get_size()
            s = min(ww / W, wh / H)
            scaled = pygame.transform.scale(canvas, (int(W * s), int(H * s)))
            window.fill(theme.BG_BOT)
            window.blit(scaled, ((ww - W * s) / 2, (wh - H * s) / 2))
            pygame.display.flip()
            if ctx.fullscreen != applied_fullscreen:
                window = _apply_fullscreen(pygame, ctx.fullscreen)
                applied_fullscreen = ctx.fullscreen
    finally:
        ctx.lobby.stop()
        close_camera(ctx)
        pygame.quit()
    return 0
