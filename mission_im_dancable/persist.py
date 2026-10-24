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
) -> list[dict]:
    """Record a run and return the updated top-score list (best first)."""
    scores = load_scores()
    scores.append(
        {
            "video": video,
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
