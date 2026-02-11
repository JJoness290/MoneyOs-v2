from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.config import OUTPUT_DIR


@dataclass(frozen=True)
class DirectorMemory:
    recent_topics: list[str]
    recent_shots: list[str]
    recent_clips: list[str]


def load_memory() -> DirectorMemory:
    path = OUTPUT_DIR / "director_history.json"
    if not path.exists():
        return DirectorMemory(recent_topics=[], recent_shots=[], recent_clips=[])
    payload = json.loads(path.read_text(encoding="utf-8"))
    return DirectorMemory(
        recent_topics=payload.get("recent_topics", []),
        recent_shots=payload.get("recent_shots", []),
        recent_clips=payload.get("recent_clips", []),
    )


def save_memory(memory: DirectorMemory) -> None:
    path = OUTPUT_DIR / "director_history.json"
    payload = {
        "recent_topics": memory.recent_topics,
        "recent_shots": memory.recent_shots,
        "recent_clips": memory.recent_clips,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
