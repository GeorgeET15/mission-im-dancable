"""UI-independent rhythm-game session: score, combo, and mission progression.

The session owns NO camera, video, or display objects. The caller feeds it
pose evaluations and elapsed time; it quantizes them into fixed scoring
intervals so points cannot scale with render FPS or duplicate frames.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from .moves import Move, move_index_at


class GamePhase(enum.Enum):
    BRIEFING = "briefing"          # move intro, waits for player ready
    ACTIVE = "active"              # performing the move once through
    GRACE = "grace"                # move time over but failing: last-chance window
    RESULT = "result"              # move finished: passed / retry
    COMPLETE = "complete"          # mission finished: scorecard


class Judgement(enum.Enum):
    PERFECT = "PERFECT"
    GREAT = "GREAT"
    MISS = "MISS"


def judge(
    accuracy: float | None,
    reliable: bool,
    great: float = 0.6,
    perfect: float = 0.85,
) -> Judgement:
    """Map one committed evaluation to a rhythm-game judgement."""
    if not reliable or accuracy is None or accuracy < great:
        return Judgement.MISS
    return Judgement.PERFECT if accuracy >= perfect else Judgement.GREAT


def grade_for(mean_accuracy: float) -> str:
    """Playful grade letter for a finished run."""
    for cutoff, letter in ((0.90, "S"), (0.75, "A"), (0.60, "B"), (0.45, "C")):
        if mean_accuracy >= cutoff:
            return letter
    return "F"


@dataclass(frozen=True)
class GameConfig:
    eval_interval: float = 0.25    # seconds between committed evaluations
    acc_threshold: float = 0.6     # accuracy needed for an eval to count
    pass_threshold: float = 0.6    # mean accuracy needed to pass a move
    min_reliable_fraction: float = 0.5  # share of evals that must be reliable
    base_points: int = 100         # points per successful evaluation
    combo_step: int = 8            # combo count per +1 multiplier
    max_multiplier: int = 5
    grace_seconds: float = 5.0     # extra time to recover a failing move (0 = off)

    def __post_init__(self) -> None:
        if self.eval_interval <= 0:
            raise ValueError("eval_interval must be positive")
        if not 0 < self.acc_threshold <= 1:
            raise ValueError("acc_threshold must be in (0, 1]")
        if not 0 <= self.pass_threshold <= 1:
            raise ValueError("pass_threshold must be in [0, 1]")
        if not 0 <= self.min_reliable_fraction <= 1:
            raise ValueError("min_reliable_fraction must be in [0, 1]")
        if self.base_points <= 0 or self.combo_step <= 0 or self.max_multiplier <= 0:
            raise ValueError("points/combo_step/max_multiplier must be positive")
        if self.grace_seconds < 0:
            raise ValueError("grace_seconds must be >= 0")


@dataclass(frozen=True)
class MoveResult:
    name: str
    passed: bool
    evaluations: int
    reliable_evaluations: int
    mean_accuracy: float
    points: int
    best_combo: int

    @property
    def reliable_fraction(self) -> float:
        return (
            self.reliable_evaluations / self.evaluations
            if self.evaluations
            else 0.0
        )


@dataclass(frozen=True)
class EvalEvent:
    """Returned when update() commits a scoring evaluation, else None."""

    hit: bool
    accuracy: float | None
    points: int
    combo: int


@dataclass
class _MoveAccum:
    evaluations: int = 0
    reliable: int = 0
    accuracy_sum: float = 0.0
    points: int = 0
    best_combo: int = 0


def _passes(acc: _MoveAccum, config: GameConfig) -> bool:
    """Shared pass rule: enough reliable evals AND good mean accuracy."""
    if acc.evaluations == 0:
        return False
    mean = acc.accuracy_sum / acc.reliable if acc.reliable else 0.0
    return (
        acc.reliable / acc.evaluations >= config.min_reliable_fraction
        and mean >= config.pass_threshold
    )


def _result_of(move: Move, acc: _MoveAccum, config: GameConfig) -> MoveResult:
    return MoveResult(
        name=move.name,
        passed=_passes(acc, config),
        evaluations=acc.evaluations,
        reliable_evaluations=acc.reliable,
        mean_accuracy=acc.accuracy_sum / acc.reliable if acc.reliable else 0.0,
        points=acc.points,
        best_combo=acc.best_combo,
    )


class GameSession:
    """Mission progression over a list of moves.

    Drive it by calling update() once per render frame with the current
    smoothed accuracy. Points are only committed once per eval_interval of
    move time, and repeated calls carrying the same frame_id are ignored.
    """

    def __init__(self, moves: list[Move], config: GameConfig | None = None) -> None:
        if not moves:
            raise ValueError("GameSession needs at least one move")
        self.config = config or GameConfig()
        self.moves = moves

        self.phase = GamePhase.BRIEFING
        self.move_index = 0
        self.time = moves[0].start
        self.paused = False

        self.total_score = 0
        self.combo = 0
        self.best_combo = 0
        self.current_accuracy: float | None = None

        self.results: list[MoveResult] = []
        self.last_result: MoveResult | None = None

        self._acc = _MoveAccum()
        self._last_now: float | None = None
        self._last_frame_id: int | None = None
        self._last_interval = -1
        self._eval_elapsed = 0.0
        self._grace_elapsed = 0.0

    @property
    def move(self) -> Move:
        return self.moves[self.move_index]

    @property
    def move_progress(self) -> float:
        """0-1 progress through the active move's time range."""
        if self.move.duration <= 0:
            return 1.0
        return min(max((self.time - self.move.start) / self.move.duration, 0.0), 1.0)

    @property
    def grace_remaining(self) -> float:
        """Seconds left in the completion window; 0 outside GRACE."""
        if self.phase != GamePhase.GRACE:
            return 0.0
        return max(0.0, self.config.grace_seconds - self._grace_elapsed)

    # -- state transitions -------------------------------------------------

    def start_move(self) -> None:
        """BRIEFING → ACTIVE (player confirmed ready)."""
        if self.phase != GamePhase.BRIEFING:
            return
        self.phase = GamePhase.ACTIVE
        self.time = self.move.start
        self._acc = _MoveAccum()
        self._last_interval = -1
        self._last_now = None
        self._eval_elapsed = 0.0
        self._grace_elapsed = 0.0

    def continue_from_result(self) -> None:
        """RESULT → next BRIEFING or COMPLETE. Ignored if the move failed."""
        if self.phase != GamePhase.RESULT or (
            self.last_result and not self.last_result.passed
        ):
            return
        if self.move_index + 1 < len(self.moves):
            self.move_index += 1
            self.phase = GamePhase.BRIEFING
            self.time = self.move.start
        else:
            self.phase = GamePhase.COMPLETE

    def retry_move(self) -> None:
        """RESULT(failed) → BRIEFING for the same move, accumulators reset."""
        if self.phase == GamePhase.RESULT and self.last_result and not self.last_result.passed:
            self.phase = GamePhase.BRIEFING
            self.time = self.move.start
            self._acc = _MoveAccum()
            self._last_interval = -1
            self._eval_elapsed = 0.0
            self._grace_elapsed = 0.0

    def toggle_pause(self) -> None:
        if self.phase in (GamePhase.ACTIVE, GamePhase.GRACE):
            self.paused = not self.paused

    # -- per-frame update ---------------------------------------------------

    def update(
        self,
        now: float,
        accuracy: float | None,
        reliable: bool,
        frame_id: int,
    ) -> EvalEvent | None:
        """Advance the session and maybe commit one scoring evaluation.

        Returns an EvalEvent when a new scoring interval was committed,
        otherwise None — including for duplicate frame_ids and repeated
        calls inside the same interval.
        """
        if accuracy is not None:
            self.current_accuracy = accuracy

        if self.phase not in (GamePhase.ACTIVE, GamePhase.GRACE) or self.paused:
            self._last_now = None
            return None

        dt = 0.0 if self._last_now is None else now - self._last_now
        self._last_now = now
        self._eval_elapsed += dt
        if self.phase == GamePhase.ACTIVE:
            self.time += dt
        else:  # GRACE: loop the move's range while the player finishes
            self._grace_elapsed += dt
            dur = self.move.duration or 1.0
            self.time = self.move.start + (self.time - self.move.start + dt) % dur

        if frame_id == self._last_frame_id:
            return None  # same camera frame re-processed — not a new event
        self._last_frame_id = frame_id

        interval = int(self._eval_elapsed / self.config.eval_interval)
        if interval == self._last_interval:
            return None
        self._last_interval = interval

        hit = reliable and accuracy is not None and accuracy >= self.config.acc_threshold
        self._acc.evaluations += 1
        points = 0
        if hit:
            self._acc.reliable += 1
            self._acc.accuracy_sum += accuracy
            points = self._points_for_eval()
            self._acc.points += points
            self.total_score += points
            self.combo += 1
            self.best_combo = max(self.best_combo, self.combo)
            self._acc.best_combo = max(self._acc.best_combo, self.combo)
        else:
            self.combo = 0

        if self.phase == GamePhase.ACTIVE and self.time >= self.move.end:
            if self._would_pass() or self.config.grace_seconds <= 0:
                self._finish_move()
            else:  # failing at move end → last-chance window, not instant fail
                self.phase = GamePhase.GRACE
                self._grace_elapsed = 0.0
                self.time = self.move.start  # replay the move from the top
        elif self.phase == GamePhase.GRACE:
            if self._would_pass() or self._grace_elapsed >= self.config.grace_seconds:
                self._finish_move()

        return EvalEvent(hit=hit, accuracy=accuracy, points=points, combo=self.combo)

    def _points_for_eval(self) -> int:
        multiplier = min(1 + self.combo // self.config.combo_step, self.config.max_multiplier)
        return self.config.base_points * multiplier

    def _would_pass(self) -> bool:
        return _passes(self._acc, self.config)

    def _finish_move(self) -> None:
        self.last_result = _result_of(self.move, self._acc, self.config)
        self.results.append(self.last_result)
        self.phase = GamePhase.RESULT
        self.paused = False

    # -- scorecard ----------------------------------------------------------

    def scorecard(self) -> dict:
        total_evals = sum(r.evaluations for r in self.results)
        total_reliable = sum(r.reliable_evaluations for r in self.results)
        acc_sum = sum(r.mean_accuracy * r.reliable_evaluations for r in self.results)
        return {
            "total_score": self.total_score,
            "best_combo": self.best_combo,
            "moves_passed": sum(1 for r in self.results if r.passed),
            "moves_total": len(self.moves),
            "mean_accuracy": acc_sum / total_reliable if total_reliable else 0.0,
            "evaluations": total_evals,
            "reliable_fraction": total_reliable / total_evals if total_evals else 0.0,
        }


class PlaySession:
    """Continuous scored run through the whole dance — the Play Mode session.

    Unlike GameSession (which pauses between moves for briefings and lets a
    failing move retry), a PlaySession runs the reference from the first
    move's start to the last move's end without interruption, like a real
    rhythm game. Scoring uses the same rules: fixed-interval evaluations,
    100 pts × combo multiplier, misses break combo. Each evaluation is
    bucketed into the segment owning its timestamp so per-move stats still
    come out.
    """

    def __init__(self, moves: list[Move], config: GameConfig | None = None) -> None:
        if not moves:
            raise ValueError("PlaySession needs at least one move")
        self.config = config or GameConfig()
        self.moves = moves
        self.start_time = moves[0].start
        self.end_time = moves[-1].end
        self.restart()

    def restart(self) -> None:
        """Reset the whole run back to the dance's start."""
        self.phase = GamePhase.ACTIVE
        self.time = self.start_time
        self.paused = False
        self.total_score = 0
        self.combo = 0
        self.best_combo = 0
        self.current_accuracy: float | None = None
        self._accs = [_MoveAccum() for _ in self.moves]
        self._last_now: float | None = None
        self._last_frame_id: int | None = None
        self._last_interval = -1
        self._eval_elapsed = 0.0

    @property
    def move_index(self) -> int:
        return move_index_at(self.moves, self.time)

    @property
    def move(self) -> Move:
        return self.moves[self.move_index]

    @property
    def move_progress(self) -> float:
        """0-1 progress through the segment under the playhead."""
        move = self.move
        if move.duration <= 0:
            return 1.0
        return min(max((self.time - move.start) / move.duration, 0.0), 1.0)

    @property
    def progress(self) -> float:
        """0-1 progress through the whole run."""
        span = self.end_time - self.start_time
        if span <= 0:
            return 1.0
        return min(max((self.time - self.start_time) / span, 0.0), 1.0)

    @property
    def results(self) -> list[MoveResult]:
        """Per-move stats so far (or final, once COMPLETE)."""
        return [
            _result_of(m, acc, self.config)
            for m, acc in zip(self.moves, self._accs)
        ]

    def toggle_pause(self) -> None:
        if self.phase == GamePhase.ACTIVE:
            self.paused = not self.paused

    def update(
        self,
        now: float,
        accuracy: float | None,
        reliable: bool,
        frame_id: int,
    ) -> EvalEvent | None:
        """Same contract as GameSession.update — one EvalEvent per committed
        scoring interval, None otherwise."""
        if accuracy is not None:
            self.current_accuracy = accuracy

        if self.phase != GamePhase.ACTIVE or self.paused:
            self._last_now = None
            return None

        dt = 0.0 if self._last_now is None else now - self._last_now
        self._last_now = now
        self._eval_elapsed += dt
        self.time = min(self.time + dt, self.end_time)
        if self.time >= self.end_time:
            # the run ends exactly at the last segment's end — immediately,
            # not at the next eval boundary
            self.phase = GamePhase.COMPLETE
            self.paused = False

        if frame_id == self._last_frame_id:
            return None  # same camera frame re-processed — not a new event
        self._last_frame_id = frame_id

        interval = int(self._eval_elapsed / self.config.eval_interval)
        if interval == self._last_interval:
            return None
        self._last_interval = interval

        hit = reliable and accuracy is not None and accuracy >= self.config.acc_threshold
        acc = self._accs[self.move_index]
        acc.evaluations += 1
        points = 0
        if hit:
            acc.reliable += 1
            acc.accuracy_sum += accuracy
            points = self._points_for_eval()
            acc.points += points
            self.total_score += points
            self.combo += 1
            self.best_combo = max(self.best_combo, self.combo)
            acc.best_combo = max(acc.best_combo, self.combo)
        else:
            self.combo = 0

        return EvalEvent(hit=hit, accuracy=accuracy, points=points, combo=self.combo)

    def _points_for_eval(self) -> int:
        multiplier = min(1 + self.combo // self.config.combo_step, self.config.max_multiplier)
        return self.config.base_points * multiplier

    def scorecard(self) -> dict:
        results = self.results
        total_evals = sum(r.evaluations for r in results)
        total_reliable = sum(r.reliable_evaluations for r in results)
        acc_sum = sum(r.mean_accuracy * r.reliable_evaluations for r in results)
        return {
            "total_score": self.total_score,
            "best_combo": self.best_combo,
            "moves_passed": sum(1 for r in results if r.passed),
            "moves_total": len(self.moves),
            "mean_accuracy": acc_sum / total_reliable if total_reliable else 0.0,
            "evaluations": total_evals,
            "reliable_fraction": total_reliable / total_evals if total_evals else 0.0,
        }
