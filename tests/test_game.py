import pytest

from mission_im_dancable.game import (
    GameConfig,
    GamePhase,
    GameSession,
)
from mission_im_dancable.moves import Move


def _moves():
    return [Move("alpha", 0.0, 1.0), Move("beta", 2.0, 3.0)]


def _session(**cfg):
    return GameSession(_moves(), GameConfig(**cfg))


def _run_active(session, n_frames=12, dt=0.1, accuracy=0.9, reliable=True, start=0.0):
    """Feed dt-spaced updates during ACTIVE; returns last EvalEvent."""
    event = None
    for i in range(n_frames):
        event = session.update(start + i * dt, accuracy, reliable, frame_id=i)
    return event


def test_config_validation():
    with pytest.raises(ValueError):
        GameConfig(eval_interval=0)
    with pytest.raises(ValueError):
        GameConfig(acc_threshold=1.5)
    with pytest.raises(ValueError):
        GameConfig(base_points=0)


def test_starts_in_briefing_no_scoring():
    s = _session()
    assert s.phase == GamePhase.BRIEFING
    assert s.update(0.0, 0.9, True, frame_id=0) is None
    assert s._acc.evaluations == 0
    assert s.total_score == 0


def test_interval_gating_limits_evals():
    s = _session()
    s.start_move()
    _run_active(s)
    # 1s move at 0.25s intervals → 5 committed evaluations despite 12 calls
    assert s._acc.evaluations == 5
    assert s.total_score == 500  # 5 evals x 100 pts at x1 multiplier


def test_score_not_scaled_by_fps():
    for fps in (0.05, 0.02):  # simulate 20fps and 50fps update rates
        s = _session()
        s.start_move()
        t = 0.0
        while s.phase == GamePhase.ACTIVE:
            s.update(t, 0.9, True, frame_id=int(t * 1000))
            t += fps
        assert s._acc.evaluations == 5
        assert s.total_score == 500


def test_duplicate_frames_are_not_events():
    s = _session()
    s.start_move()
    s.update(0.0, 0.9, True, frame_id=7)
    event = s.update(0.3, 0.9, True, frame_id=7)  # same frame re-sent
    assert event is None
    assert s._acc.evaluations == 1


def test_combo_grows_and_breaks():
    s = _session()
    s.start_move()
    s.update(0.0, 0.9, True, frame_id=0)
    s.update(0.3, 0.9, True, frame_id=1)
    assert s.combo == 2
    s.update(0.6, 0.2, True, frame_id=2)  # accurate tracking, bad pose → miss
    assert s.combo == 0
    s.update(0.9, 0.9, False, frame_id=3)  # unreliable → miss
    assert s.combo == 0


def test_multiplier_increases_with_combo():
    s = _session(combo_step=2, base_points=100)
    s.start_move()
    s.update(0.0, 0.9, True, frame_id=0)   # combo 0→1, x1
    assert s.total_score == 100
    s.update(0.3, 0.9, True, frame_id=1)   # combo 1→2, still x1
    assert s.total_score == 200
    s.update(0.6, 0.9, True, frame_id=2)   # combo 2→3, x2
    assert s.total_score == 400


def test_move_pass_and_progression():
    s = _session()
    s.start_move()
    _run_active(s)
    assert s.phase == GamePhase.RESULT
    assert s.last_result.passed
    assert s.last_result.mean_accuracy == pytest.approx(0.9)
    s.continue_from_result()
    assert s.phase == GamePhase.BRIEFING
    assert s.move_index == 1


def test_failed_move_must_retry_not_skip():
    s = _session()
    s.start_move()
    _run_active(s, accuracy=0.1)  # everything misses
    assert s.phase == GamePhase.RESULT
    assert not s.last_result.passed
    s.continue_from_result()  # cannot advance on a failed move
    assert s.phase == GamePhase.RESULT
    assert s.move_index == 0
    s.retry_move()
    assert s.phase == GamePhase.BRIEFING
    assert s._acc.evaluations == 0
    assert s.time == s.move.start


def test_fully_unreliable_move_fails():
    s = _session()
    s.start_move()
    _run_active(s, accuracy=None, reliable=False)
    assert s.last_result is not None
    assert not s.last_result.passed
    assert s.last_result.reliable_evaluations == 0


def test_mission_complete_and_scorecard():
    s = _session()
    for _ in _moves():
        s.start_move()
        _run_active(s)
        s.continue_from_result()
    assert s.phase == GamePhase.COMPLETE
    card = s.scorecard()
    # move 1: 5 evals x100 at x1 = 500; move 2: combo 5→10 crosses the
    # x2 multiplier at combo 8 → 3x100 + 2x200 = 700; total 1200
    assert card["total_score"] == 1200
    assert card["moves_passed"] == 2
    assert card["moves_total"] == 2
    assert card["mean_accuracy"] == pytest.approx(0.9)
    assert card["best_combo"] == 10


def test_pause_freezes_time_and_evals():
    s = _session()
    s.start_move()
    s.update(0.0, 0.9, True, frame_id=0)
    s.toggle_pause()
    s.update(0.5, 0.9, True, frame_id=1)
    s.update(1.0, 0.9, True, frame_id=2)
    assert s._acc.evaluations == 1
    assert s.time == pytest.approx(0.0)
    s.toggle_pause()
    s.update(1.1, 0.9, True, frame_id=3)   # first tick after unpause: dt resets
    s.update(1.4, 0.9, True, frame_id=4)   # time advances 0.3 → new interval
    assert s._acc.evaluations == 2


def test_retry_resets_interval_state():
    s = _session()
    s.start_move()
    _run_active(s, accuracy=0.1)
    s.retry_move()
    s.start_move()
    _run_active(s, accuracy=0.95)
    assert s._acc.evaluations == 5
    assert s.last_result is not None and s.last_result.passed
