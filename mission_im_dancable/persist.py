"""Local persistence for settings and high scores.

Data lives in user directories (XDG_CONFIG_HOME / XDG_DATA_HOME, or
~/.config and ~/.local/share) — never in the repository.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

APP_DIRNAME = "mission-im-dancable"
MAX_SCORES = 10

DEFAULT_SETTINGS = {
    "camera_index": 0,
    "mirror": True,
    "last_video": "",
    "fullscreen": False,
    "window_s": 0.15,
    "volume": 80,
    "player_name": "PLAYER",
}


def config_dir() -> Path:
    base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / APP_DIRNAME


def data_dir() -> Path:
    base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / APP_DIRNAME


def _read_json(path: Path, fallback):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return fallback


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def load_settings() -> dict:
    stored = _read_json(config_dir() / "settings.json", {})
    settings = dict(DEFAULT_SETTINGS)
    if isinstance(stored, dict):
        settings.update({k: stored[k] for k in DEFAULT_SETTINGS if k in stored})
    return settings


def save_settings(settings: dict) -> None:
    _write_json(config_dir() / "settings.json", settings)


def load_scores() -> list[dict]:
    scores = _read_json(data_dir() / "highscores.json", [])
    return scores if isinstance(scores, list) else []


def add_score(
    video: str,
    score: int,
    mean_accuracy: float,
    best_combo: int,
    grade: str,
    mode: str = "performance",
    name: str = "",
) -> list[dict]:
    """Record a run and return the updated top-score list (best first)."""
    scores = load_scores()
    scores.append(
        {
            "video": video,
            "name": name.strip()[:16] or "PLAYER",
            "score": int(score),
            "mean_accuracy": round(float(mean_accuracy), 3),
            "best_combo": int(best_combo),
            "grade": grade,
            "mode": mode,
            "ts": int(time.time()),
        }
    )
    scores.sort(key=lambda s: s["score"], reverse=True)
    scores = scores[:MAX_SCORES]
    _write_json(data_dir() / "highscores.json", scores)
    return scores


# -- Learn Mode progress -----------------------------------------------------

# Move statuses are marked by the player, never inferred from pose data —
# only the dancer knows when a move is actually mastered.
MOVE_STATUSES = ("learning", "practicing", "mastered")


def _progress_path() -> Path:
    return data_dir() / "progress.json"


def load_progress() -> dict:
    """{video_name: {"moves": {segment_index_str: status}}} — {} when absent."""
    progress = _read_json(_progress_path(), {})
    return progress if isinstance(progress, dict) else {}


def save_progress(progress: dict) -> None:
    _write_json(_progress_path(), progress)


def move_statuses(video: str) -> dict[int, str]:
    """{segment_index: status} for one dance; invalid entries are dropped."""
    raw = load_progress().get(video, {})
    raw = raw.get("moves", {}) if isinstance(raw, dict) else {}
    out: dict[int, str] = {}
    if isinstance(raw, dict):
        for key, status in raw.items():
            try:
                idx = int(key)
            except (TypeError, ValueError):
                continue
            if status in MOVE_STATUSES:
                out[idx] = status
    return out


def set_move_status(video: str, index: int, status: str | None) -> dict[int, str]:
    """Mark a segment learning/practicing/mastered (None clears). Returns the
    video's updated status map."""
    progress = load_progress()
    entry = progress.setdefault(video, {"moves": {}})
    moves = entry.setdefault("moves", {})
    if status is None:
        moves.pop(str(index), None)
    elif status in MOVE_STATUSES:
        moves[str(index)] = status
    else:
        raise ValueError(f"status must be one of {MOVE_STATUSES} or None")
    if not moves:
        progress.pop(video, None)  # nothing left to remember for this dance
    save_progress(progress)
    return move_statuses(video)
