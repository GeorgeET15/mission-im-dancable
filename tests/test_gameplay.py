"""Headless gameplay tests: PlayScreen.update driven with fakes."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mission_im_dancable.config import AppConfig
from mission_im_dancable.game import GamePhase, GameSession
from mission_im_dancable.moves import Move
from mission_im_dancable.pose import PoseResult
from mission_im_dancable.practice import PracticeSession
from mission_im_dancable.reference import ReferenceChoreography
from mission_im_dancable.ui.screens import PlayScreen, ResultsScreen


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


def _ctx(mode="performance", live=True):
    moves = [Move("m1", 0.0, 1.0)]
    return SimpleNamespace(
        cfg=AppConfig(window_s=0.15, confidence=0.3),
        window_s=lambda: 0.15,
        mode=mode,
        choreo=_choreo(),
        moves=moves,
        mirror=False,
        tracker=FakeTracker(live),
        settings={"camera_index": 0},
        cap=None,
        auto_start=False,
        fullscreen=False,
        particles=SimpleNamespace(burst=lambda *a, **k: None, draw=lambda *a: None, update=lambda dt: None),
        manager=FakeManager(),
        t=0.0,
        last_session=None,
        last_results=None,
    )


def _screen(ctx, cap_ok=True):
    scr = PlayScreen()
    scr.enter()
    scr._ready = True
    scr.cap = FakeCap(ok=cap_ok)
    scr.session = (
        GameSession(ctx.moves) if ctx.mode == "performance"
        else PracticeSession(ctx.choreo, ctx.moves, ctx.cfg.window_s)
    )
    return scr


def test_practice_advances_by_dt():
    ctx = _ctx(mode="practice")
    scr = _screen(ctx)
    scr.update(ctx, 0.3)
    assert scr.session.time == pytest.approx(0.3)
    scr.paused = True
    scr.update(ctx, 0.5)
    assert scr.session.time == pytest.approx(0.3)  # frozen while paused


def test_performance_evals_commit_with_time(monkeypatch):
    ctx = _ctx()
    scr = _screen(ctx)
    scr.session.start_move()
    wall = [0.0]
    monkeypatch.setattr("time.perf_counter", lambda: wall[0])
    for _ in range(6):
        wall[0] += 0.2
        scr.update(ctx, 0.2)
    # ~1.2s of play at 0.25s intervals → ~4-5 committed evals, all hits
    assert scr.session._acc.evaluations >= 4
    assert scr.session.combo >= 4
    assert scr.session.total_score > 0


def test_unreliable_live_pose_breaks_combo(monkeypatch):
    ctx = _ctx(live=False)
    scr = _screen(ctx)
    scr.session.start_move()
    wall = [0.0]
    monkeypatch.setattr("time.perf_counter", lambda: wall[0])
    for _ in range(4):
        wall[0] += 0.3
        scr.update(ctx, 0.3)
    assert scr.session._acc.evaluations >= 3
    assert scr.session._acc.reliable == 0
    assert scr.session.combo == 0
    assert scr.session.total_score == 0


def test_camera_failure_is_miss_not_crash(monkeypatch):
    ctx = _ctx()
    scr = _screen(ctx, cap_ok=False)
    scr.session.start_move()
    wall = [0.0]
    monkeypatch.setattr("time.perf_counter", lambda: wall[0])
    for _ in range(3):
        wall[0] += 0.3
        scr.update(ctx, 0.3)
    assert scr.session._acc.reliable == 0  # all misses, no exception


def test_paused_session_commits_nothing(monkeypatch):
    ctx = _ctx()
    scr = _screen(ctx)
    scr.session.start_move()
    scr.session.toggle_pause()
    wall = [0.0]
    monkeypatch.setattr("time.perf_counter", lambda: wall[0])
    for _ in range(4):
        wall[0] += 0.5
        scr.update(ctx, 0.5)
    assert scr.session._acc.evaluations == 0


def test_complete_moves_to_results(monkeypatch):
    ctx = _ctx()
    scr = _screen(ctx)
    scr.session.start_move()
    wall = [0.0]
    monkeypatch.setattr("time.perf_counter", lambda: wall[0])
    for _ in range(8):
        wall[0] += 0.3
        scr.update(ctx, 0.3)
    # move finished → RESULT; player continues (SPACE) → COMPLETE
    assert scr.session.phase == GamePhase.RESULT
    scr.session.continue_from_result()
    assert scr.session.phase == GamePhase.COMPLETE
    scr.update(ctx, 0.1)  # session→results handoff happens next update
    assert ctx.last_session is scr.session
    assert ctx.last_results == scr.session.results
    assert isinstance(ctx.manager.replaced, ResultsScreen)
