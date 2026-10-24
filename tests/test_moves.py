import json

import pytest

from mission_im_dancable.moves import Move, MovesFileError, default_moves, load_moves


def _write(tmp_path, data) -> str:
    path = tmp_path / "moves.json"
    path.write_text(data if isinstance(data, str) else json.dumps(data))
    return path


def test_load_moves_valid(tmp_path):
    path = _write(
        tmp_path,
        {"moves": [{"name": "sway", "start": 0.0, "end": 2.5}, {"name": "spin", "start": 2.5, "end": 5.0}]},
    )
    moves = load_moves(path, video_duration=10.0)
    assert moves == [Move("sway", 0.0, 2.5), Move("spin", 2.5, 5.0)]
    assert moves[0].duration == pytest.approx(2.5)


def test_load_moves_malformed_json(tmp_path):
    path = _write(tmp_path, "{not json")
    with pytest.raises(MovesFileError, match="not valid JSON"):
        load_moves(path, 10.0)


def test_load_moves_missing_file(tmp_path):
    with pytest.raises(MovesFileError, match="Cannot read"):
        load_moves(tmp_path / "absent.json", 10.0)


def test_load_moves_requires_moves_list(tmp_path):
    for bad in [{}, {"moves": []}, {"moves": "x"}, [1, 2]]:
        path = _write(tmp_path, bad)
        with pytest.raises(MovesFileError, match="'moves' list"):
            load_moves(path, 10.0)


def test_load_moves_field_validation(tmp_path):
    cases = [
        ({"moves": [{"start": 0, "end": 1}]}, "name"),
        ({"moves": [{"name": "", "start": 0, "end": 1}]}, "name"),
        ({"moves": [{"name": "a", "end": 1}]}, "start"),
        ({"moves": [{"name": "a", "start": "0", "end": 1}]}, "numeric 'start'"),
        ({"moves": [{"name": "a", "start": 0}]}, "'end'"),
        ({"moves": ["just-a-string"]}, "must be an object"),
    ]
    for data, message in cases:
        path = _write(tmp_path, data)
        with pytest.raises(MovesFileError, match=message):
            load_moves(path, 10.0)


def test_load_moves_timestamp_range(tmp_path):
    path = _write(tmp_path, {"moves": [{"name": "a", "start": -1.0, "end": 2.0}]})
    with pytest.raises(MovesFileError, match="before the video begins"):
        load_moves(path, 10.0)

    path = _write(tmp_path, {"moves": [{"name": "a", "start": 5.0, "end": 5.0}]})
    with pytest.raises(MovesFileError, match="must be after start"):
        load_moves(path, 10.0)

    path = _write(tmp_path, {"moves": [{"name": "a", "start": 5.0, "end": 4.0}]})
    with pytest.raises(MovesFileError, match="must be after start"):
        load_moves(path, 10.0)

    path = _write(tmp_path, {"moves": [{"name": "a", "start": 0.0, "end": 11.0}]})
    with pytest.raises(MovesFileError, match="exceeds the video duration"):
        load_moves(path, 10.0)


def test_end_clamped_to_duration(tmp_path):
    path = _write(tmp_path, {"moves": [{"name": "a", "start": 9.9, "end": 10.0005}]})
    moves = load_moves(path, 10.0)
    assert moves[0].end == 10.0


def test_default_moves_full_video():
    moves = default_moves(12.5)
    assert moves == [Move("full choreography", 0.0, 12.5)]
