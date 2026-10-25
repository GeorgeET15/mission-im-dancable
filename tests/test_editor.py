"""Segment editor: draft edits, validation, save/reload, auto-suggestions."""

import json
from pathlib import Path

import numpy as np
import pytest

from mission_im_dancable.beats import BeatGrid
from mission_im_dancable.editor import SegmentDraft, draft_moves, suggest_boundaries
from mission_im_dancable.moves import (
    Move,
    load_dance_file,
    save_dance_file,
    validate_segments,
)
from mission_im_dancable.reference import ReferenceChoreography


def _choreo(moving=True):
    """12 samples/s over 10 s; moving poses produce high motion energy."""
    n = 100
    k = np.zeros((n, 17, 3), dtype=np.float32)
    k[:, :, :2] = 100.0
    k[:, :, 2] = 0.9
    if moving:
        k[:, :, 0] += np.arange(n)[:, None] * 5.0  # steady horizontal motion
    return ReferenceChoreography(
        video_path=Path("v.mp4"), video_fps=30.0, frame_count=300,
        width=640, height=480, sample_fps=10.0,
        timestamps=np.arange(n) / 10.0, keypoints=k, model_name="m",
    )


# -- draft edits -----------------------------------------------------------------


def test_split_at_playhead():
    d = SegmentDraft([Move("a", 0.0, 10.0)], 10.0)
    assert d.split_at(4.0)
    assert [(m.start, m.end) for m in d.segments] == [(0.0, 4.0), (4.0, 10.0)]
    assert d.dirty


def test_split_rejects_too_thin_pieces():
    d = SegmentDraft([Move("a", 0.0, 1.0)], 1.0)
    assert not d.split_at(0.05)
    assert len(d.segments) == 1
    d2 = SegmentDraft([Move("a", 0.0, 10.0)], 10.0)
    assert not d2.split_at(50.0)  # outside any segment → no-op


def test_merge_next_removes_boundary():
    d = SegmentDraft([Move("a", 0.0, 2.0), Move("b", 2.0, 5.0)], 5.0)
    assert d.merge_next()
    assert d.segments == [Move("a", 0.0, 5.0)]
    assert not d.merge_next()  # nothing left to merge


def test_nudge_contiguous_boundary():
    d = SegmentDraft([Move("a", 0.0, 2.0), Move("b", 2.0, 5.0)], 5.0)
    assert d.nudge_boundary(0.5)
    assert (d.segments[0].end, d.segments[1].start) == (2.5, 2.5)
    for _ in range(30):
        d.nudge_boundary(-0.1)
    assert d.segments[0].duration > 0.15  # never collapses below MIN_SEGMENT


def test_nudge_gap_moves_only_end():
    d = SegmentDraft([Move("a", 0.0, 2.0), Move("b", 3.0, 5.0)], 5.0)
    assert d.nudge_boundary(0.5)
    assert d.segments[0].end == 2.5 and d.segments[1].start == 3.0


def test_reorder_and_rename():
    d = SegmentDraft([Move("a", 0.0, 1.0), Move("b", 1.0, 2.0)], 2.0)
    d.rename("first")
    assert d.segments[0].name == "first"
    assert d.reorder(1)
    assert d.segments[0].name == "b" and d.selected == 1
    assert d.errors()  # swapped ranges now overlap → save is blocked


def test_validation_catches_bad_segments():
    assert validate_segments([], 10.0)
    assert validate_segments([Move("a", -1, 1)], 10.0)
    assert validate_segments([Move("a", 1, 1)], 10.0)
    assert validate_segments([Move("a", 0, 5), Move("b", 4, 6)], 10.0)
    assert validate_segments([Move("a", 0, 20)], 10.0)
    assert validate_segments([Move("", 0, 5)], 10.0)
    assert not validate_segments([Move("a", 0, 5), Move("b", 5, 10)], 10.0)


def test_accept_and_dirty_flags():
    d = SegmentDraft([Move("a", 0, 1)], 1.0)
    d.set_segments([Move("p1", 0, 1)], from_draft=True)
    assert d.from_draft and d.dirty
    d.accept()
    assert not d.from_draft


# -- save / reload ------------------------------------------------------------------


def test_save_and_reload_roundtrip(tmp_path):
    target = tmp_path / "v.moves.json"
    moves = [Move("a", 0.0, 2.0), Move("b", 2.0, 5.0, "hit it")]
    save_dance_file(target, moves)
    loaded, meta = load_dance_file(target, 5.0)
    assert loaded == moves
    assert meta == {}


def test_save_preserves_existing_metadata(tmp_path):
    target = tmp_path / "v.moves.json"
    target.write_text(json.dumps({
        "id": "x", "title": "My Title", "difficulty": "hard",
        "moves": [{"name": "old", "start": 0.0, "end": 1.0}],
    }))
    save_dance_file(target, [Move("new", 0.0, 2.0)])
    raw = json.loads(target.read_text())
    assert raw["title"] == "My Title" and raw["difficulty"] == "hard"
    assert raw["moves"][0]["name"] == "new"


def test_load_invalid_sidecar_still_errors(tmp_path):
    bad = tmp_path / "v.moves.json"
    bad.write_text('{"moves": [{"name": "a", "start": 9.0, "end": 99.0}]}')
    with pytest.raises(Exception, match="exceeds the video duration"):
        load_dance_file(bad, 10.0)


# -- suggestions ----------------------------------------------------------------------


def test_suggest_from_beat_grid():
    beats = BeatGrid(np.arange(0.0, 12.0, 0.5))  # 120 BPM → a beat every 0.5 s
    bounds = suggest_boundaries(_choreo(), beats, 12.0)
    assert bounds == [3.5, 7.5]  # every 8th beat inside the dance
    moves = draft_moves(12.0, bounds)
    assert len(moves) == 3 and moves[0].name == "phrase 1"


def test_suggest_from_pose_motion_when_no_beats():
    # alternating still / moving blocks → valleys between bursts of motion
    ch = _choreo()
    n = ch.keypoints.shape[0]
    mask = (np.arange(n) // 20) % 2 == 0  # 2 s moving, 2 s still
    ch.keypoints[mask, :, 0] += np.arange(n)[mask, None] * 8.0
    bounds = suggest_boundaries(ch, BeatGrid(np.array([])), 10.0)
    assert bounds
    assert all(0.5 < b < 9.5 for b in bounds)


def test_suggest_empty_when_nothing_works():
    flat = _choreo()
    flat.keypoints[:, :, 2] = 0.0  # no detections at all
    assert suggest_boundaries(flat, BeatGrid(np.array([])), 10.0) == []


def test_draft_moves_never_zero_length():
    moves = draft_moves(10.0, [3.0, 3.0, 7.0])
    assert not validate_segments(moves, 10.0)


# -- LearnScreen integration ---------------------------------------------------------


def test_learn_edit_mode_flow(tmp_path, monkeypatch):
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("XDG_DATA_HOME", str(tmp_path / "data"))
    import pygame
    from types import SimpleNamespace
    from mission_im_dancable.config import AppConfig
    from mission_im_dancable.pose import PoseResult
    from mission_im_dancable.ui.learn import LearnScreen

    video = tmp_path / "v.mp4"  # no sidecar file — default segment path
    ctx = SimpleNamespace(
        cfg=AppConfig(window_s=0.15, confidence=0.3),
        window_s=lambda: 0.15,
        mode="learn",
        choreo=_choreo(),
        moves=[Move("full choreography", 0.0, 10.0)],
        dance=None,
        video_path=video,
        moves_path=None,
        mirror=False,
        beats=BeatGrid(np.arange(0.0, 10.0, 0.5)),
        tracker=SimpleNamespace(predict=lambda f: None),
        settings={}, cap=None, manager=SimpleNamespace(pop=lambda: None, depth=2),
        t=0.0, quit=False,
        particles=SimpleNamespace(burst=lambda *a, **k: None, draw=lambda *a: None, update=lambda dt: None),
    )
    scr = LearnScreen()
    scr.enter()
    scr.update(ctx, 0.016)

    key = lambda k, u="": pygame.event.Event(pygame.KEYDOWN, key=k, unicode=u)
    scr.handle_event(ctx, key(pygame.K_e))           # enter edit mode
    assert scr.editing and len(scr.draft.segments) == 1  # default segment
    scr.session.seek(4.0)
    scr.handle_event(ctx, key(pygame.K_b))           # split at playhead
    assert len(scr.draft.segments) == 2
    scr.handle_event(ctx, key(pygame.K_g))           # auto-draft (confirm first)
    scr.handle_event(ctx, key(pygame.K_g))
    assert scr.draft.from_draft
    scr.handle_event(ctx, key(pygame.K_a))           # accept suggestions
    assert not scr.draft.from_draft
    scr.handle_event(ctx, key(pygame.K_s))           # save — no sidecar yet → writes
    assert (tmp_path / "v.moves.json").is_file()
    assert ctx.moves_path == tmp_path / "v.moves.json"
    assert len(ctx.moves) == len(scr.draft.segments)
    scr.handle_event(ctx, key(pygame.K_e))           # leave edit mode (clean)
    assert not scr.editing


def test_start_auto_segments_without_sidecar(tmp_path, monkeypatch):
    """A video with no .moves.json gets drafted steps at _start, not one blob."""
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    from types import SimpleNamespace
    from mission_im_dancable.config import AppConfig
    from mission_im_dancable.ui.screens import SelectScreen

    pushed = []
    ctx = SimpleNamespace(
        cfg=AppConfig(window_s=0.15, confidence=0.3),
        window_s=lambda: 0.15, mode="learn",
        video_path=tmp_path / "v.mp4", moves_path=None,
        choreo=_choreo(), beats=BeatGrid(np.arange(0.0, 10.0, 0.5)),
        mirror=False, settings={}, manager=SimpleNamespace(push=pushed.append),
        choreo_done=None,
    )
    scr = SelectScreen()
    scr.enter()
    scr.choreo = ctx.choreo
    scr.moves = [Move("full choreography", 0.0, 10.0)]
    scr._start(ctx)
    assert len(ctx.moves) > 1                      # auto-draft gave real steps
    assert ctx.moves is scr.moves
    assert ctx.dance is not None and pushed        # and Learn got pushed


def test_start_keeps_annotated_segments(tmp_path, monkeypatch):
    """A real sidecar file's segments are never overwritten by a draft."""
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    from types import SimpleNamespace
    from mission_im_dancable.config import AppConfig
    from mission_im_dancable.ui.screens import SelectScreen

    sidecar = tmp_path / "v.moves.json"
    sidecar.write_text(json.dumps({"moves": [{"name": "real", "start": 0.0, "end": 5.0},
                                             {"name": "real2", "start": 5.0, "end": 10.0}]}))
    ctx = SimpleNamespace(
        cfg=AppConfig(window_s=0.15, confidence=0.3),
        window_s=lambda: 0.15, mode="learn",
        video_path=tmp_path / "v.mp4", moves_path=sidecar,
        choreo=_choreo(), beats=BeatGrid(np.arange(0.0, 10.0, 0.5)),
        mirror=False, settings={}, manager=SimpleNamespace(push=lambda s: None),
    )
    scr = SelectScreen()
    scr.enter()
    scr.choreo = ctx.choreo
    scr.moves = [Move("real", 0.0, 5.0), Move("real2", 5.0, 10.0)]
    scr._start(ctx)
    assert ctx.moves == scr.moves                  # annotations untouched


def test_save_requires_confirm_when_file_exists(tmp_path, monkeypatch):
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("XDG_DATA_HOME", str(tmp_path / "data"))
    import pygame
    from types import SimpleNamespace
    from mission_im_dancable.config import AppConfig
    from mission_im_dancable.ui.learn import LearnScreen

    video = tmp_path / "v.mp4"
    sidecar = tmp_path / "v.moves.json"
    sidecar.write_text(json.dumps({"moves": [{"name": "old", "start": 0.0, "end": 10.0}]}))
    ctx = SimpleNamespace(
        cfg=AppConfig(window_s=0.15, confidence=0.3),
        window_s=lambda: 0.15, mode="learn", choreo=_choreo(),
        moves=[Move("old", 0.0, 10.0)], dance=None,
        video_path=video, moves_path=sidecar,
        mirror=False, beats=None,
        tracker=SimpleNamespace(predict=lambda f: None),
        settings={}, cap=None, manager=SimpleNamespace(pop=lambda: None, depth=2),
        t=0.0, quit=False,
        particles=SimpleNamespace(burst=lambda *a, **k: None, draw=lambda *a: None, update=lambda dt: None),
    )
    scr = LearnScreen()
    scr.enter()
    scr.update(ctx, 0.016)
    key = lambda k, u="": pygame.event.Event(pygame.KEYDOWN, key=k, unicode=u)
    scr.handle_event(ctx, key(pygame.K_e))
    scr.session.seek(3.0)
    scr.handle_event(ctx, key(pygame.K_b))
    before = sidecar.read_text()
    scr.handle_event(ctx, key(pygame.K_s))   # first press → asks for confirm
    assert "overwrite" in scr._warn
    assert sidecar.read_text() == before     # not written yet
    scr.handle_event(ctx, key(pygame.K_s))   # confirmed → writes
    assert sidecar.read_text() != before
