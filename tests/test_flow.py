"""Flow tests: navigation swaps, ready/countdown, auto-start, tracking loss."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # headless surfaces

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pygame
import pytest

from mission_im_dancable.config import AppConfig
from mission_im_dancable.game import GamePhase, GameSession, PlaySession
from mission_im_dancable.moves import Move
from mission_im_dancable.pose import PoseResult
from mission_im_dancable.reference import ReferenceChoreography
from mission_im_dancable.ui.screens import PlayScreen, ReadyScreen, ResultsScreen, SelectScreen


def _pose(conf=0.9):
    k = np.zeros((17, 3))
    k[:, :2] = 100.0
    k[:, 2] = conf
    return k


def _choreo(n=20):
    k = np.zeros((n, 17, 3), dtype=np.float32)
    k[:, :, :2] = 100.0
    k[:, :, 2] = 0.9
    return ReferenceChoreography(
        video_path=Path("v.mp4"), video_fps=30.0, frame_count=60,
        width=640, height=480, sample_fps=10.0,
        timestamps=np.arange(n) / 10.0, keypoints=k, model_name="m",
    )


class FakeCap:
    def __init__(self, ok=True):
        self.ok = ok
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.released = False

    def read(self):
        return (self.ok, self.frame) if self.ok else (False, None)

    def release(self):
        self.released = True


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


def _ctx(live=True):
    moves = [Move("m1", 0.0, 1.0)]
    return SimpleNamespace(
        cfg=AppConfig(window_s=0.15, confidence=0.3),
        window_s=lambda: 0.15,
        mode="play",
        choreo=_choreo(),
        moves=moves,
        video_path=Path("v.mp4"),
        mirror=False,
        tracker=FakeTracker(live),
        settings={"camera_index": 0},
        cap=None,
        auto_start=False,
        fullscreen=False,
        particles=SimpleNamespace(burst=lambda *a, **k: None, draw=lambda *a: None, update=lambda dt: None),
        manager=FakeManager(),
        t=0.0,
        quit=False,
        last_session=None,
        last_results=None,
    )


def _play_screen(ctx, cap_ok=True):
    scr = PlayScreen()
    scr.enter()
    scr._ready = True
    scr.cap = FakeCap(ok=cap_ok)
    scr.session = PlaySession(ctx.moves)
    return scr


# -- navigator.swap ----------------------------------------------------------


def test_nav_swap_replaces_top_without_entering_below():
    from mission_im_dancable.ui.nav import Navigator

    nav = Navigator(fade=0.0)
    calls = {"a": 0, "b": 0, "c": 0}

    class S:
        def __init__(self, key):
            self.key = key
        def enter(self):
            calls[self.key] += 1
        def exit(self):
            pass

    a, b, c = S("a"), S("b"), S("c")
    nav.push(a)
    nav.push(b)
    nav.swap(c)
    assert nav.current is c and nav.depth == 2
    assert calls["a"] == 1  # 'a' was NOT re-entered between b and c
    assert calls["b"] == 1
    nav.pop()
    assert nav.current is a and calls["a"] == 2


# -- ReadyScreen -------------------------------------------------------------


def test_ready_detect_countdown_finish():
    ctx = _ctx(live=True)
    ctx.cap = FakeCap()
    scr = ReadyScreen()
    scr.enter()
    scr.update(ctx, 0.016)            # opening → detect (cap already present)
    assert scr.phase == "detect"
    for _ in range(int(ReadyScreen.DETECT_HOLD / 0.016) + 2):
        scr.update(ctx, 0.016)
    assert scr.phase == "locked"
    for _ in range(int(ReadyScreen.LOCK_BEAT / 0.016) + 2):
        scr.update(ctx, 0.016)
    assert scr.phase == "countdown"
    for _ in range(int(ReadyScreen.COUNTDOWN / 0.05) + 2):
        scr.update(ctx, 0.05)
    assert isinstance(ctx.manager.swapped, PlayScreen)
    assert ctx.auto_start is True


def test_ready_countdown_pauses_when_pose_lost():
    ctx = _ctx(live=True)
    ctx.cap = FakeCap()
    scr = ReadyScreen()
    scr.enter()
    scr.update(ctx, 0.016)
    for _ in range(int(ReadyScreen.DETECT_HOLD / 0.05) + 2):
        scr.update(ctx, 0.05)
    for _ in range(int(ReadyScreen.LOCK_BEAT / 0.05) + 2):
        scr.update(ctx, 0.05)
    assert scr.phase == "countdown"
    t0 = scr.count_t
    ctx.tracker.live = False
    for _ in range(4):
        scr.update(ctx, 0.1)
    assert scr.count_t == pytest.approx(t0)  # frozen while pose is gone
    for _ in range(int(ReadyScreen.LOST_TIMEOUT / 0.1) + 2):
        scr.update(ctx, 0.1)
    assert scr.phase == "detect"  # aborted back to detection


def test_ready_error_and_back():
    ctx = _ctx(live=True)
    scr = ReadyScreen()
    scr.enter()
    scr.err = "no camera"
    scr.update(ctx, 0.016)
    assert scr.phase == "error"
    scr._go_back(ctx)
    assert ctx.manager.popped == 1 and scr._canceled


def test_ready_back_during_opening_releases_late_camera():
    ctx = _ctx()
    scr = ReadyScreen()
    scr.enter()
    scr.update(ctx, 0.016)           # thread starts
    scr._go_back(ctx)                # user bails mid-open
    # simulate the worker thread finishing after the pop
    ctx.cap = FakeCap()
    if scr._canceled:
        from mission_im_dancable.ui.screens import close_camera
        close_camera(ctx)
    assert ctx.cap is None


# -- PlayScreen extras --------------------------------------------------------


def test_play_starts_active_after_countdown(monkeypatch):
    ctx = _ctx()
    ctx.auto_start = True
    ctx.cap = FakeCap()
    scr = PlayScreen()
    scr.enter()
    scr.update(ctx, 0.016)
    assert scr.session.phase == GamePhase.ACTIVE  # continuous run, no briefing


def test_play_no_miss_popups_when_pose_lost(monkeypatch):
    ctx = _ctx(live=False)
    scr = _play_screen(ctx)
    wall = [0.0]
    monkeypatch.setattr("time.perf_counter", lambda: wall[0])
    for _ in range(4):
        wall[0] += 0.3
        scr.update(ctx, 0.3)
    assert scr.session._accs[0].evaluations >= 3  # evals still committed (scoring intact)
    assert scr.popups == []                        # but no MISS spam
    assert scr.no_pose_t > 0                       # tracking-loss chip is armed


def test_play_camera_error_r_retries_via_ready():
    ctx = _ctx()
    scr = _play_screen(ctx, cap_ok=False)
    scr.cam_err = "boom"
    scr.cap = None
    ev = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r)
    scr.handle_event(ctx, ev)
    assert isinstance(ctx.manager.swapped, ReadyScreen)


def test_play_q_pops_when_stacked():
    ctx = _ctx()
    ctx.manager.depth = 3
    scr = _play_screen(ctx)
    ev = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_q)
    scr.handle_event(ctx, ev)
    assert ctx.manager.popped == 1


def test_play_q_replaces_select_at_depth_one():
    ctx = _ctx()
    scr = _play_screen(ctx)
    ev = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_q)
    scr.handle_event(ctx, ev)
    assert isinstance(ctx.manager.replaced, SelectScreen)


# -- ResultsScreen ------------------------------------------------------------


def _results_ctx(tmp_path, monkeypatch, scores=()):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    ctx = _ctx()
    ctx.last_session = GameSession(ctx.moves)
    ctx.last_session._acc.accuracy_sum = 0.8
    ctx.last_session._acc.reliable = 1
    ctx.last_session._acc.evaluations = 1
    ctx.last_session.results = []
    ctx.last_results = []
    for score in scores:
        from mission_im_dancable.persist import add_score
        add_score("v.mp4", score, 0.5, 1, "C")
    return ctx


def test_results_new_best_first_run(tmp_path, monkeypatch):
    ctx = _results_ctx(tmp_path, monkeypatch)
    scr = ResultsScreen()
    scr.enter()
    scr.update(ctx, 0.016)
    assert scr.new_best is True and scr.card is not None


def test_results_below_best_no_badge(tmp_path, monkeypatch):
    ctx = _results_ctx(tmp_path, monkeypatch, scores=[99999])
    scr = ResultsScreen()
    scr.enter()
    scr.update(ctx, 0.016)
    assert scr.new_best is False


# -- draw-after-swap regression -------------------------------------------------


def _draw_ctx():
    """ctx with real fonts + canvas so draw() actually executes."""
    pygame.font.init()
    ctx = _ctx()
    ctx.fonts = {
        "sm": pygame.font.SysFont("dejavusansmono", 18),
        "med": pygame.font.SysFont("dejavusansmono", 24, bold=True),
        "big": pygame.font.SysFont("dejavusansmono", 56, bold=True),
        "title": pygame.font.SysFont("dejavusansmono", 64, bold=True),
    }
    ctx.bg = pygame.Surface((1280, 720))
    ctx.beats = None
    return ctx


def test_play_draw_after_mid_update_swap():
    """PlayScreen entered via swap() during update() must draw before its
    first update() — regression for session=None crash."""
    ctx = _draw_ctx()
    surf = pygame.Surface((1280, 720))
    scr = PlayScreen()
    scr.enter()
    scr.draw(ctx, surf)  # no update() call first
    assert scr.session is not None


def test_ready_draw_states():
    ctx = _draw_ctx()
    ctx.cap = FakeCap()
    surf = pygame.Surface((1280, 720))
    scr = ReadyScreen()
    scr.enter()
    scr.draw(ctx, surf)  # opening
    scr.update(ctx, 0.016)
    scr.draw(ctx, surf)  # detect
    scr.err = "boom"; scr.phase = "error"
    scr.draw(ctx, surf)  # error
    scr.phase = "countdown"; scr.count_t = 2.0
    scr.draw(ctx, surf)  # countdown


def test_results_draw_before_sync():
    ctx = _draw_ctx()
    surf = pygame.Surface((1280, 720))
    scr = ResultsScreen()
    scr.enter()
    scr.draw(ctx, surf)  # no card yet → "no results yet" path, must not crash


# -- BrowseScreen ---------------------------------------------------------------


def test_browse_navigates_and_picks(tmp_path):
    from mission_im_dancable.ui.screens import BrowseScreen

    (tmp_path / "sub").mkdir()
    (tmp_path / "clip.mp4").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("no")
    ctx = _ctx()
    ctx.browse_result = None
    scr = BrowseScreen()
    scr.enter()
    scr.cwd = tmp_path
    scr._refresh()
    names = [n for _, n, _ in scr.entries]
    assert "..  (up)" in names and "sub/" in names and "clip.mp4" in names
    assert "notes.txt" not in names  # non-video files hidden

    # navigate into subdir
    di = names.index("sub/")
    scr._activate_row(ctx, di)
    assert scr.cwd == tmp_path / "sub"
    # back up via ".."
    scr._activate_row(ctx, 0)
    assert scr.cwd == tmp_path
    # pick the video → browse_result + pop
    vi = [n for _, n, _ in scr.entries].index("clip.mp4")
    scr._activate_row(ctx, vi)
    assert ctx.browse_result == tmp_path / "clip.mp4"
    assert ctx.manager.popped == 1


def test_browse_cancel_pops():
    from mission_im_dancable.ui.screens import BrowseScreen

    ctx = _ctx()
    scr = BrowseScreen()
    scr.enter()
    scr.cwd = Path(".")
    scr._refresh()
    scr.handle_event(ctx, pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
    assert ctx.manager.popped == 1


def test_browse_probe_never_raises_and_caches(tmp_path):
    from mission_im_dancable.ui.screens import BrowseScreen

    bad = tmp_path / "fake.mp4"
    bad.write_bytes(b"not a video")
    scr = BrowseScreen()
    scr.enter()
    info = scr._probe(bad)
    assert info["size"] == len(b"not a video") and info["dur"] == 0.0
    assert scr._probe(bad) is info  # cached — one probe per file


def test_browse_click_selects_then_opens(tmp_path):
    from mission_im_dancable.ui.screens import BrowseScreen

    (tmp_path / "a.mp4").write_bytes(b"x")
    ctx = _ctx()
    ctx.browse_result = None
    scr = BrowseScreen()
    scr.enter()
    scr.cwd = tmp_path
    scr._refresh()
    vi = [n for _, n, _ in scr.entries].index("a.mp4")
    scr.sel = 0
    y = scr.LIST_Y + vi * scr.ROW_H + 4
    click = pygame.event.Event(pygame.MOUSEBUTTONDOWN,
                               pos=(scr.LIST_X + 5, y), button=1)
    scr.handle_event(ctx, click)
    assert scr.sel == vi and ctx.browse_result is None  # first click selects
    scr.handle_event(ctx, click)
    assert ctx.browse_result == tmp_path / "a.mp4"      # second click opens


def test_browse_thumb_cache_debounced(tmp_path):
    from mission_im_dancable.ui.screens import BrowseScreen

    (tmp_path / "a.mp4").write_bytes(b"x")
    ctx = _ctx()
    scr = BrowseScreen()
    scr.enter()
    scr.cwd = tmp_path
    scr._refresh()
    vi = [n for _, n, _ in scr.entries].index("a.mp4")
    scr.sel = vi
    scr._thumb_wait = 0.05
    scr.update(ctx, 0.1)   # debounce fires → probe attempted (fails gracefully)
    assert scr._thumb_sel == vi
    assert tmp_path / "a.mp4" in scr._thumbs  # cached even when unreadable


# -- soundtrack wiring ---------------------------------------------------------


class FakeMusic:
    def __init__(self):
        self.calls = []
        self._pos = None

    @property
    def loaded(self):
        return any(c[0] == "load" and c[1] for c in self.calls)

    def load(self, path):
        self.calls.append(("load", path is not None))
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

    def set_volume(self, v):
        self.calls.append(("volume", v))


def test_play_music_follows_run(monkeypatch):
    ctx = _ctx()
    ctx.music = FakeMusic()
    ctx.audio_path = Path("x.wav")
    scr = _play_screen(ctx)
    wall = [0.0]
    monkeypatch.setattr("time.perf_counter", lambda: wall[0])
    scr.update(ctx, 0.016)  # ACTIVE from the first frame — music starts at 0
    assert ("load", True) in ctx.music.calls
    assert ("play", 0.0) in ctx.music.calls
    scr.session.toggle_pause()
    scr.update(ctx, 0.1)
    assert ("pause",) in ctx.music.calls
    scr.session.toggle_pause()
    wall[0] += 0.1
    scr.update(ctx, 0.1)
    assert ("unpause",) in ctx.music.calls
    # run to the end of the dance → COMPLETE → music stops
    for _ in range(20):
        wall[0] += 0.3
        scr.update(ctx, 0.3)
        if isinstance(ctx.manager.replaced, ResultsScreen):
            break
    assert scr.session.phase == GamePhase.COMPLETE
    assert ("stop",) in ctx.music.calls


def test_music_disabled_is_silent(monkeypatch):
    from mission_im_dancable.audio import Music

    m = Music(enabled=False)
    assert m.load(Path("x.wav")) is False
    m.play_at(0.0); m.pause(); m.unpause(); m.stop()  # all no-ops
    assert m.position() is None and m.loaded is False


def test_watch_music_starts_and_stops():
    ctx = _ctx()
    ctx.music = FakeMusic()
    ctx.audio_path = Path("x.wav")
    from mission_im_dancable.ui.screens import WatchScreen

    scr = WatchScreen()
    scr.enter()
    scr.update(ctx, 0.016)   # clock created + music starts at 0
    assert ("play", 0.0) in ctx.music.calls
    scr.handle_event(ctx, pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
    scr.update(ctx, 0.016)
    assert ("pause",) in ctx.music.calls
    scr.handle_event(ctx, pygame.event.Event(pygame.KEYDOWN, key=pygame.K_q))
    scr.exit()
    assert ("stop",) in ctx.music.calls
