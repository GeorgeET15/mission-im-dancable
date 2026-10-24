import json

import pytest

from mission_im_dancable.game import Judgement, grade_for, judge
from mission_im_dancable.persist import (
    DEFAULT_SETTINGS,
    add_score,
    load_scores,
    load_settings,
    save_settings,
)
from mission_im_dancable.ui.nav import Navigator


class FakeScreen:
    def __init__(self):
        self.calls = []

    def enter(self):
        self.calls.append("enter")

    def exit(self):
        self.calls.append("exit")

    def handle_event(self, ctx, ev):
        self.calls.append(("event", ev))

    def update(self, ctx, dt):
        self.calls.append(("update", dt))

    def draw(self, ctx, surf):
        self.calls.append("draw")


def test_nav_push_pop_replace():
    nav = Navigator(fade=0.0)
    a, b, c = FakeScreen(), FakeScreen(), FakeScreen()
    nav.push(a)
    assert nav.current is a and a.calls == ["enter"]
    nav.push(b)
    assert a.calls == ["enter", "exit"] and nav.current is b
    old = nav.pop()
    assert old is b and nav.current is a and a.calls[-1] == "enter"
    nav.replace(c)
    assert nav.depth == 1 and nav.current is c and a.calls[-1] == "exit"


def test_nav_delegates_events():
    nav = Navigator(fade=0.0)
    s = FakeScreen()
    nav.push(s)
    nav.handle_event(None, "e1")
    nav.update(None, 0.016)
    nav.draw(None, None)
    assert ("event", "e1") in s.calls and ("update", 0.016) in s.calls


def test_nav_input_locked_during_fade():
    nav = Navigator(fade=0.5)
    s = FakeScreen()
    nav.push(s)
    nav.handle_event(None, "e1")  # ignored: transition active
    nav.update(None, 0.3)          # still fading
    nav.handle_event(None, "e2")  # ignored
    nav.update(None, 0.3)          # fade over
    nav.handle_event(None, "e3")
    kinds = [c for c in s.calls if isinstance(c, tuple) and c[0] == "event"]
    assert kinds == [("event", "e3")]


def test_judge_thresholds():
    assert judge(0.95, True) == Judgement.PERFECT
    assert judge(0.85, True) == Judgement.PERFECT
    assert judge(0.7, True) == Judgement.GREAT
    assert judge(0.6, True) == Judgement.GREAT
    assert judge(0.59, True) == Judgement.MISS
    assert judge(0.9, False) == Judgement.MISS
    assert judge(None, True) == Judgement.MISS


def test_grade_letters():
    assert grade_for(0.95) == "S"
    assert grade_for(0.8) == "A"
    assert grade_for(0.65) == "B"
    assert grade_for(0.5) == "C"
    assert grade_for(0.1) == "F"


def test_settings_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    s = load_settings()
    assert s == DEFAULT_SETTINGS
    s["camera_index"] = 2
    s["mirror"] = False
    s["amy"] = False
    save_settings(s)
    assert load_settings()["camera_index"] == 2
    assert load_settings()["mirror"] is False
    assert load_settings()["amy"] is False


def test_scores_sorted_and_capped(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    for i in range(15):
        add_score("v.mp4", score=i * 100, mean_accuracy=0.5, best_combo=i, grade="B")
    scores = load_scores()
    assert len(scores) == 10
    assert scores[0]["score"] == 1400 and scores[-1]["score"] == 500


def test_scores_corrupt_file_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    p = tmp_path / "data" / "mission-im-dancable"
    p.mkdir(parents=True)
    (p / "highscores.json").write_text("{broken")
    assert load_scores() == []
