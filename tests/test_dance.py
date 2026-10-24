"""Dance annotation files: metadata, instructions, invalid input."""

import json

import pytest

from mission_im_dancable.dance import build_dance
from mission_im_dancable.moves import MovesFileError, load_dance_file, load_moves
from mission_im_dancable.reference import ReferenceChoreography

import numpy as np
from pathlib import Path


def _write(tmp_path, data):
    path = tmp_path / "d.moves.json"
    path.write_text(data if isinstance(data, str) else json.dumps(data))
    return path


def _choreo():
    return ReferenceChoreography(
        video_path=Path("v.mp4"), video_fps=30.0, frame_count=300,
        width=640, height=480, sample_fps=10.0,
        timestamps=np.arange(100) / 10.0,
        keypoints=np.zeros((100, 17, 3), dtype=np.float32), model_name="m",
    )


def test_dance_file_full_metadata(tmp_path):
    path = _write(tmp_path, {
        "id": "routine-7",
        "title": "Neon Routine",
        "difficulty": "hard",
        "moves": [
            {"name": "sway", "start": 0.0, "end": 2.0, "instruction": "shift weight"},
            {"name": "spin", "start": 2.0, "end": 4.0},
        ],
    })
    moves, meta = load_dance_file(path, 10.0)
    assert meta == {"id": "routine-7", "title": "Neon Routine", "difficulty": "hard"}
    assert moves[0].instruction == "shift weight"
    assert moves[1].instruction == ""  # optional field


def test_dance_file_without_meta_still_loads(tmp_path):
    path = _write(tmp_path, {"moves": [{"name": "a", "start": 0.0, "end": 1.0}]})
    moves, meta = load_dance_file(path, 5.0)
    assert len(moves) == 1 and meta == {}


def test_dance_file_bad_meta_ignored_not_fatal(tmp_path):
    path = _write(tmp_path, {
        "title": 42, "difficulty": ["x"],
        "moves": [{"name": "a", "start": 0.0, "end": 1.0}],
    })
    moves, meta = load_dance_file(path, 5.0)
    assert len(moves) == 1 and meta == {}


def test_dance_file_invalid_move_still_errors(tmp_path):
    path = _write(tmp_path, {
        "title": "fine",
        "moves": [{"name": "a", "start": 3.0, "end": 1.0}],
    })
    with pytest.raises(MovesFileError, match="must be after start"):
        load_dance_file(path, 5.0)


def test_load_moves_still_returns_just_moves(tmp_path):
    path = _write(tmp_path, {
        "title": "t",
        "moves": [{"name": "a", "start": 0.0, "end": 1.0}],
    })
    moves = load_moves(path, 5.0)
    assert moves[0].name == "a"


def test_non_string_instruction_ignored(tmp_path):
    path = _write(tmp_path, {
        "moves": [{"name": "a", "start": 0.0, "end": 1.0, "instruction": 7}],
    })
    moves, _ = load_dance_file(path, 5.0)
    assert moves[0].instruction == ""


def test_build_dance_uses_meta_then_fallbacks():
    from mission_im_dancable.moves import Move

    choreo = _choreo()
    mv = [Move("only", 0.0, 10.0)]
    d = build_dance(Path("clip.mp4"), choreo, mv, {"title": "Show", "difficulty": "easy"})
    assert d.id == "clip" and d.title == "Show" and d.difficulty == "easy"
    assert d.duration == pytest.approx(10.0) and d.video_fps == 30.0
    assert d.moves is mv
    d2 = build_dance(Path("clip.mp4"), choreo, mv, {})
    assert d2.title == "clip" and d2.difficulty == ""
