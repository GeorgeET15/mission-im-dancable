"""LearnSession: free-form study transport — seek, loop, speed, segments."""

import pytest

from mission_im_dancable.learn import LearnSession, SPEEDS
from mission_im_dancable.moves import Move

MOVES = [Move("a", 0.0, 2.0), Move("b", 2.0, 5.0), Move("c", 5.0, 10.0)]


def _session():
    return LearnSession(MOVES, duration=10.0, video_fps=30.0)


def test_seek_clamps_to_dance():
    s = _session()
    s.seek(7.5)
    assert s.time == 7.5
    s.seek(-3)
    assert s.time == 0.0
    s.seek(99)
    assert s.time == 10.0


def test_play_pause_and_replay():
    s = _session()
    s.toggle()
    assert s.playing
    s.toggle()
    assert not s.playing
    s.seek(10.0)  # at the very end
    s.toggle()    # replay from the top
    assert s.playing and s.time == 0.0


def test_advance_respects_speed():
    s = _session()
    s.toggle()
    s.advance(1.0)
    assert s.time == pytest.approx(1.0)
    s.set_speed(0.5)
    s.advance(1.0)
    assert s.time == pytest.approx(1.5)


def test_pause_stops_advance():
    s = _session()
    s.advance(1.0)
    assert s.time == 0.0


def test_end_of_dance_stops_playing():
    s = _session()
    s.toggle()
    s.advance(20.0)
    assert not s.playing and s.time == 10.0 and s.finished


def test_frame_stepping_pauses_and_steps():
    s = _session()
    s.toggle()
    s.step_frames(3)  # 3 frames at 30 fps = 0.1 s
    assert not s.playing
    assert s.time == pytest.approx(0.1)
    s.step_frames(-3)
    assert s.time == pytest.approx(0.0)


def test_segment_selection_and_navigation():
    s = _session()
    assert s.select(1) == 1
    assert s.time == 2.0  # jumped to the segment start
    assert s.next_segment() == 2 and s.time == 5.0
    assert s.prev_segment() == 1 and s.time == 2.0
    s.select(-5)
    assert s.selected == 0
    s.select(99)
    assert s.selected == 2


def test_segment_loop_wraps_exactly():
    s = _session()
    s.select(1)
    s.cycle_loop()
    assert s.loop == "segment"
    s.toggle()
    s.advance(3.5)  # past seg end 5.0? no: 2.0 + 3.5 = 5.5 → wraps to 2.5
    assert s.time == pytest.approx(2.5)
    for _ in range(10):
        s.advance(1.0)
    assert 2.0 <= s.time <= 5.0  # never escapes the segment


def test_loop_all_wraps_dance():
    s = _session()
    s.cycle_loop()
    s.cycle_loop()
    assert s.loop == "all"
    s.toggle()
    s.advance(11.0)
    assert s.time == pytest.approx(1.0)
    assert s.playing


def test_loop_off_at_end_stops():
    s = _session()
    s.select(1)
    s.toggle()
    s.advance(10.0)  # plays past the segment, stops at dance end
    assert s.time == 10.0 and not s.playing


def test_cycle_loop_and_speed():
    s = _session()
    assert [s.cycle_loop() for _ in range(3)] == ["segment", "all", "off"]
    s.set_speed(0.6)
    assert s.speed == 0.5
    while s.speed != 1.0:
        s.cycle_speed(1)
    s.cycle_speed(1)
    assert s.speed == SPEEDS[0]  # wraps around


def test_active_index_and_segment_state():
    s = _session()
    s.select(1)
    s.seek(6.0)
    assert s.active_index == 2
    assert s.segment_state(0) == "done"       # played through already
    assert s.segment_state(1) == "selected"   # selection outranks its state
    assert s.segment_state(2) == "active"     # under the playhead
    s.select(0)
    assert s.segment_state(2) == "upcoming"   # not done, not selected, not active


def test_gap_belongs_to_upcoming_segment():
    s = LearnSession([Move("a", 0.0, 1.0), Move("b", 2.0, 3.0)], duration=3.0)
    s.seek(1.5)
    assert s.active_index == 1


def test_invalid_construction():
    with pytest.raises(ValueError):
        LearnSession([], 10.0)
    with pytest.raises(ValueError):
        LearnSession(MOVES, 0.0)
