"""PlaySession: continuous scored run through the whole dance."""

import pytest

from mission_im_dancable.game import GamePhase, PlaySession
from mission_im_dancable.moves import Move

MOVES = [Move("a", 0.0, 2.0), Move("b", 2.0, 4.0), Move("c", 4.0, 6.0)]


def _session():
    return PlaySession(MOVES)


def _run(s, seconds, dt=0.3, acc=0.9, frame=[0]):
    """Feed the session wall-clock updates; returns committed EvalEvents."""
    now = [0.0]
    events = []
    for _ in range(int(seconds / dt)):
        now[0] += dt
        frame[0] += 1
        ev = s.update(now[0], acc, True, frame[0])
        if ev is not None:
            events.append(ev)
    return events


def test_starts_active_at_first_move():
    s = _session()
    assert s.phase == GamePhase.ACTIVE
    assert s.time == 0.0 and s.move_index == 0


def test_runs_continuously_through_segments():
    s = _session()
    _run(s, 7.0)
    assert s.phase == GamePhase.COMPLETE
    assert s.time == pytest.approx(6.0)
    # evals were bucketed across all three segments
    assert len(s.results) == 3
    assert all(r.evaluations > 0 for r in s.results)


def test_scoring_and_combo_math_unchanged():
    s = _session()
    events = _run(s, 2.5, dt=0.3)  # ~8 committed evals at 0.25s intervals
    assert len(events) >= 8
    assert all(ev.hit for ev in events)
    assert s.total_score == sum(ev.points for ev in events)
    assert s.best_combo == len(events)


def test_miss_breaks_combo():
    s = _session()
    _run(s, 1.0, acc=0.9)
    assert s.combo > 0
    _run(s, 1.0, acc=0.1)  # below the 0.6 hit threshold → misses
    assert s.combo == 0


def test_gap_time_belongs_to_upcoming_move():
    moves = [Move("a", 0.0, 1.0), Move("b", 2.0, 3.0)]
    s = PlaySession(moves)
    s.time = 1.5  # between segments
    assert s.move_index == 1


def test_pause_freezes_time_and_evals():
    s = _session()
    _run(s, 1.0)
    frozen_t = s.time
    s.toggle_pause()
    _run(s, 1.0)
    assert s.time == pytest.approx(frozen_t)
    evals = s._accs[0].evaluations
    s.toggle_pause()
    _run(s, 1.0)
    assert s._accs[0].evaluations > evals


def test_duplicate_frames_never_score():
    s = _session()
    evs = [
        s.update(0.1 * i, 0.9, True, frame_id=7)
        for i in range(1, 10)
    ]
    # same frame_id → at most one committed eval (the first interval crossing)
    assert sum(ev is not None for ev in evs) <= 1


def test_results_and_scorecard_shape():
    s = _session()
    _run(s, 6.5)
    card = s.scorecard()
    assert card["moves_total"] == 3
    assert card["evaluations"] > 0
    assert card["mean_accuracy"] == pytest.approx(0.9, abs=0.01)
    assert card["total_score"] == s.total_score
    assert card["moves_passed"] == 3


def test_restart_resets_everything():
    s = _session()
    _run(s, 3.0)
    s.restart()
    assert s.phase == GamePhase.ACTIVE
    assert s.time == 0.0 and s.total_score == 0 and s.combo == 0
    assert all(r.evaluations == 0 for r in s.results)


def test_empty_moves_rejected():
    with pytest.raises(ValueError):
        PlaySession([])


def test_progress_fraction():
    s = _session()
    s.time = 3.0
    assert s.progress == pytest.approx(0.5)
    assert s.move_index == 1
    assert s.move_progress == pytest.approx(0.5)
