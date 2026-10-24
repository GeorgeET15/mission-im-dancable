"""Per-dance learning progress persistence (Learn Mode)."""

import pytest

from mission_im_dancable.persist import (
    load_progress,
    move_statuses,
    set_move_status,
)


def test_statuses_empty_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    assert move_statuses("dance_1.mp4") == {}


def test_set_and_read_statuses(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    set_move_status("dance_1.mp4", 0, "learning")
    set_move_status("dance_1.mp4", 2, "mastered")
    set_move_status("other.mp4", 1, "practicing")
    assert move_statuses("dance_1.mp4") == {0: "learning", 2: "mastered"}
    assert move_statuses("other.mp4") == {1: "practicing"}


def test_clear_status_removes_entry(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    set_move_status("v.mp4", 0, "learning")
    out = set_move_status("v.mp4", 0, None)
    assert out == {}
    assert "v.mp4" not in load_progress()  # empty entry cleaned up


def test_invalid_status_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    with pytest.raises(ValueError, match="learning.*practicing.*mastered"):
        set_move_status("v.mp4", 0, "perfect")


def test_corrupt_progress_file_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    p = tmp_path / "data" / "mission-im-dancable"
    p.mkdir(parents=True)
    (p / "progress.json").write_text("{broken")
    assert move_statuses("v.mp4") == {}


def test_junk_entries_dropped(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    set_move_status("v.mp4", 0, "learning")
    # hand-corrupt the file with junk entries
    from mission_im_dancable.persist import save_progress
    save_progress({"v.mp4": {"moves": {"0": "learning", "x": "weird", "3": "bogus"}}})
    assert move_statuses("v.mp4") == {0: "learning"}
