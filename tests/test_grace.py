"""Grace-window tests: a failing move gets a last-chance period, not an
instant fail."""

import pytest

from mission_im_dancable.game import GameConfig, GamePhase, GameSession
from mission_im_dancable.moves import Move


def _session(grace=5.0):
    return GameSession([Move("m1", 0.0, 1.0)], GameConfig(grace_seconds=grace))


def _drive(s, now, acc, reliable, fid):
    return s.update(now, acc, reliable, fid)


def test_passing_move_skips_grace():
    s = _session()
    s.start_move()
    for i in range(6):  # 1.5s of reliable good poses → past move.end
        s.update(i * 0.25, 0.9, True, i)
    assert s.phase == GamePhase.RESULT
    assert s.last_result.passed


def test_failing_move_enters_grace_then_expires():
    s = _session()
    s.start_move()
    fid = 0
    now = 0.0
    while s.phase == GamePhase.ACTIVE:  # miss every eval until move.end
        now += 0.25
        fid += 1
        s.update(now, None, False, fid)
    assert s.phase == GamePhase.GRACE
    assert s.grace_remaining > 4.0
    assert s.time == s.move.end  # reference frozen on the last pose
    while s.phase == GamePhase.GRACE:  # keep failing → window expires
        assert s.time == s.move.end
        now += 0.25
        fid += 1
        s.update(now, None, False, fid)
    assert s.phase == GamePhase.RESULT
    assert not s.last_result.passed


def test_recovery_during_grace_passes_early():
    s = _session()
    s.start_move()
    fid = now = 0
    while s.phase == GamePhase.ACTIVE:
        now += 0.25
        fid += 1
        s.update(now, None, False, fid)
    assert s.phase == GamePhase.GRACE
    # sudden improvement inside the window → passes before it expires
    last_remaining = s.grace_remaining
    for _ in range(6):
        now += 0.25
        fid += 1
        s.update(now, 0.9, True, fid)
        if s.phase == GamePhase.GRACE:
            last_remaining = s.grace_remaining
    assert s.phase == GamePhase.RESULT
    assert last_remaining > 0  # ended early via pass, not deadline
    assert s.last_result.evaluations > 4


def test_grace_zero_preserves_old_fail_behavior():
    s = _session(grace=0.0)
    s.start_move()
    fid = now = 0
    while s.phase == GamePhase.ACTIVE:
        now += 0.25
        fid += 1
        s.update(now, None, False, fid)
    assert s.phase == GamePhase.RESULT
    assert not s.last_result.passed


def test_grace_window_pauses_and_results_continue_normally():
    s = _session()
    s.start_move()
    fid = now = 0
    while s.phase == GamePhase.ACTIVE:
        now += 0.25
        fid += 1
        s.update(now, None, False, fid)
    s.toggle_pause()
    assert s.paused and s.phase == GamePhase.GRACE
    t0 = s.time
    s.update(now + 10.0, None, False, fid + 1)  # paused → time frozen
    assert s.time == t0 and s.phase == GamePhase.GRACE
    s.toggle_pause()
    while s.phase == GamePhase.GRACE:
        now += 0.25
        fid += 1
        s.update(now, None, False, fid)
    # failed move → retry still works
    s.retry_move()
    assert s.phase == GamePhase.BRIEFING


def test_config_rejects_negative_grace():
    with pytest.raises(ValueError):
        GameConfig(grace_seconds=-1)
