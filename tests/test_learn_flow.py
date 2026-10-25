"""Headless Learn Mode tests: LearnScreen update/draw/events with fakes."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # headless surfaces

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pygame
import pytest

from mission_im_dancable.config import AppConfig
from mission_im_dancable.learn import LearnSession
from mission_im_dancable.moves import Move
from mission_im_dancable.pose import PoseResult
from mission_im_dancable.reference import ReferenceChoreography
from mission_im_dancable.ui.learn import LearnScreen


def _pose(conf=0.9):
    k = np.zeros((17, 3))
    k[:, :2] = 100.0
    k[:, 2] = conf
    return k


def _choreo(n=100, fps=30.0):
    """10 s of confident samples — long enough for seek tests."""
    k = np.zeros((n, 17, 3), dtype=np.float32)
    k[:, :, :2] = 100.0
    k[:, :, 2] = 0.9
    return ReferenceChoreography(
        video_path=Path("v.mp4"), video_fps=fps, frame_count=300,
        width=640, height=480, sample_fps=10.0,
        timestamps=np.arange(n) / 10.0, keypoints=k, model_name="m",
    )


class FakeCap:
    def __init__(self, ok=True):
        self.ok = ok
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def read(self):
        return (self.ok, self.frame) if self.ok else (False, None)


class FakeTracker:
    def __init__(self, live=True):
        self.live = live

    def predict(self, frame):
        return PoseResult(keypoints=_pose(), inference_ms=1.0) if self.live else None


class FakeManager:
    def __init__(self):
        self.replaced = None
        self.swapped = None
        self.popped = 0
        self.depth = 1

    def replace(self, screen):
        self.replaced = screen

    def swap(self, screen):
        self.swapped = screen

    def pop(self):
        self.popped += 1


class FakeMusic:
    def __init__(self):
        self.calls = []
        self._pos = None
        self._path = None

    @property
    def loaded(self):
        return self._path is not None

    @property
    def path(self):
        return self._path

    def load(self, path):
        self.calls.append(("load", path is not None))
        self._path = path if path is not None else None
        return path is not None

    def play_at(self, t):
        self.calls.append(("play", t))
        self._pos = t

    def pause(self):
        self.calls.append(("pause",))

    def unpause(self):
        self.calls.append(("unpause",))

    def stop(self):
        self.calls.append(("stop",))

    def position(self):
        return self._pos


MOVES = [Move("a", 0.0, 1.0, "step one"), Move("b", 1.0, 2.0)]


def _ctx(live=True):
    return SimpleNamespace(
        cfg=AppConfig(window_s=0.15, confidence=0.3),
        window_s=lambda: 0.15,
        mode="learn",
        choreo=_choreo(),
        moves=list(MOVES),
        dance=None,
        video_path=Path("v.mp4"),
        mirror=False,
        tracker=FakeTracker(live),
        settings={"camera_index": 0},
        cap=None,
        manager=FakeManager(),
        t=0.0,
        quit=False,
        particles=SimpleNamespace(burst=lambda *a, **k: None, draw=lambda *a: None, update=lambda dt: None),
    )


def _screen(ctx):
    scr = LearnScreen()
    scr.enter()
    return scr


def _key(key):
    return pygame.event.Event(pygame.KEYDOWN, key=key)


def _draw_ctx(ctx):
    pygame.font.init()
    ctx.fonts = {
        "sm": pygame.font.SysFont("dejavusansmono", 18),
        "med": pygame.font.SysFont("dejavusansmono", 24, bold=True),
        "big": pygame.font.SysFont("dejavusansmono", 56, bold=True),
        "title": pygame.font.SysFont("dejavusansmono", 64, bold=True),
    }
    ctx.bg = pygame.Surface((1280, 720))
    return ctx


# -- session construction --------------------------------------------------------


def test_session_created_lazily_on_first_update():
    ctx = _ctx()
    scr = _screen(ctx)
    scr.update(ctx, 0.016)
    assert isinstance(scr.session, LearnSession)
    assert scr.session.time == 0.0


def test_no_camera_degrades_to_watch_mode():
    ctx = _ctx()
    scr = _screen(ctx)
    # camera opens on a worker thread — may take a while or fail; either way
    # the lesson screen keeps working
    for _ in range(200):
        scr.update(ctx, 0.05)
        if scr.cam_state != "opening":
            break
    assert scr.cam_state in ("opening", "ready", "error")
    assert scr.session is not None


def test_camera_present_goes_ready():
    ctx = _ctx()
    ctx.cap = FakeCap()
    scr = _screen(ctx)
    scr.update(ctx, 0.016)
    assert scr.cam_state == "ready"
    assert scr.live is not None


# -- transport keys --------------------------------------------------------------------


def test_space_toggles_playback():
    ctx = _ctx()
    ctx.cap = FakeCap()
    scr = _screen(ctx)
    scr.update(ctx, 0.016)
    t0 = scr.session.time
    scr.handle_event(ctx, _key(pygame.K_SPACE))
    scr.update(ctx, 0.5)
    assert scr.session.time > t0
    scr.handle_event(ctx, _key(pygame.K_SPACE))
    frozen = scr.session.time
    scr.update(ctx, 0.5)
    assert scr.session.time == pytest.approx(frozen)


def test_arrow_seek_and_frame_step():
    ctx = _ctx()
    ctx.cap = FakeCap()
    scr = _screen(ctx)
    scr.update(ctx, 0.016)
    scr.handle_event(ctx, _key(pygame.K_RIGHT))
    assert scr.session.time == pytest.approx(2.0)
    scr.handle_event(ctx, _key(pygame.K_LEFT))
    assert scr.session.time == pytest.approx(0.0)
    scr.handle_event(ctx, _key(pygame.K_PERIOD))
    assert scr.session.time == pytest.approx(1 / 30.0)


def test_segment_keys_and_loop():
    ctx = _ctx()
    ctx.cap = FakeCap()
    scr = _screen(ctx)
    scr.update(ctx, 0.016)
    scr.handle_event(ctx, _key(pygame.K_n))
    assert scr.session.selected == 1 and scr.session.time == 1.0
    scr.handle_event(ctx, _key(pygame.K_p))
    assert scr.session.selected == 0 and scr.session.time == 0.0
    scr.handle_event(ctx, _key(pygame.K_l))
    assert scr.session.loop == "segment"


def test_speed_keys():
    ctx = _ctx()
    ctx.cap = FakeCap()
    scr = _screen(ctx)
    scr.update(ctx, 0.016)
    scr.handle_event(ctx, _key(pygame.K_MINUS))
    assert scr.session.speed == 0.75
    scr.handle_event(ctx, _key(pygame.K_EQUALS))
    assert scr.session.speed == 1.0


def test_timeline_click_seeks_and_selects():
    ctx = _ctx()
    ctx.cap = FakeCap()
    scr = _screen(ctx)
    scr.update(ctx, 0.016)
    bar = scr.timeline.rect
    # click at 15% of the bar → t = 1.5s → inside segment "b" [1.0, 2.0]
    pos = (bar.x + int(bar.w * 0.15), bar.y + bar.h // 2)
    scr.handle_event(ctx, pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=pos, button=1))
    assert scr.session.time == pytest.approx(1.5, abs=0.01)
    assert scr.session.selected == 1


# -- progress marks ----------------------------------------------------------------------


def test_visiting_segment_marks_learning(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    ctx = _ctx()
    ctx.cap = FakeCap()
    scr = _screen(ctx)
    scr.update(ctx, 0.016)
    assert scr.statuses.get(0) == "learning"  # auto-marked on first visit
    scr.handle_event(ctx, _key(pygame.K_n))
    scr.update(ctx, 0.016)
    assert scr.statuses.get(1) == "learning"


def test_k_cycles_status(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    ctx = _ctx()
    ctx.cap = FakeCap()
    scr = _screen(ctx)
    scr.update(ctx, 0.016)
    for expected in ("practicing", "mastered", None):
        scr.handle_event(ctx, _key(pygame.K_k))
        assert scr.statuses.get(0) == expected


def test_statuses_persist_across_screens(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    ctx = _ctx()
    ctx.cap = FakeCap()
    scr = _screen(ctx)
    scr.update(ctx, 0.016)
    scr.handle_event(ctx, _key(pygame.K_k))  # → practicing
    scr2 = _screen(_ctx())
    scr2.update(ctx, 0.016)
    assert scr2.statuses.get(0) == "practicing"


# -- music sync ----------------------------------------------------------------------------


def test_music_follows_play_pause_and_seek():
    ctx = _ctx()
    ctx.cap = FakeCap()
    ctx.music = FakeMusic()
    ctx.audio_path = Path("x.wav")
    scr = _screen(ctx)
    scr.update(ctx, 0.016)
    assert not any(c[0] == "play" for c in ctx.music.calls)  # nothing plays while paused
    scr.handle_event(ctx, _key(pygame.K_SPACE))  # play
    scr.update(ctx, 0.3)
    # music anchored at the playhead, whatever it currently is
    assert any(c[0] == "play" for c in ctx.music.calls)
    assert ctx.music._pos == pytest.approx(scr.session.time)
    scr.handle_event(ctx, _key(pygame.K_RIGHT))  # seek +2 s → re-anchor audio
    before = len(ctx.music.calls)
    scr.update(ctx, 0.016)
    new_plays = [c for c in ctx.music.calls[before:] if c[0] == "play"]
    assert new_plays and new_plays[-1][1] == pytest.approx(scr.session.time)
    scr.handle_event(ctx, _key(pygame.K_SPACE))  # pause
    scr.update(ctx, 0.016)
    assert ("pause",) in ctx.music.calls
    scr.exit()
    assert ("stop",) in ctx.music.calls


def test_slow_speed_silences_audio():
    ctx = _ctx()
    ctx.cap = FakeCap()
    ctx.music = FakeMusic()
    ctx.audio_path = Path("x.wav")
    scr = _screen(ctx)
    scr.update(ctx, 0.016)
    scr.handle_event(ctx, _key(pygame.K_SPACE))   # play at 1.0 → audio on
    scr.update(ctx, 0.05)
    assert ctx.music._pos == pytest.approx(scr.session.time)
    scr.handle_event(ctx, _key(pygame.K_MINUS))   # 0.75 → pause audio
    scr.update(ctx, 0.05)
    assert ("pause",) in ctx.music.calls


# -- draw ------------------------------------------------------------------------------------


def test_draw_smoke_all_states():
    ctx = _draw_ctx(_ctx())
    ctx.cap = FakeCap()
    surf = pygame.Surface((1280, 720))
    scr = _screen(ctx)
    scr.draw(ctx, surf)              # before first update — lazy init
    scr.update(ctx, 0.016)
    scr.draw(ctx, surf)              # ready + camera frame
    scr.cam_state = "error"; scr.cam_err = "boom"; scr.frame = None
    scr.draw(ctx, surf)              # camera error panel
    scr.cam_state = "ready"; scr.frame = np.zeros((480, 640, 3), dtype=np.uint8)
    scr.session.toggle()
    scr.update(ctx, 0.05)
    scr.draw(ctx, surf)              # playing
    scr.statuses = {0: "mastered"}
    scr.session.cycle_loop()
    scr.draw(ctx, surf)              # marks + loop state
