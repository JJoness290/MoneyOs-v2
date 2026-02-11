from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ShotPlan:
    shot_id: str
    shot_type: str
    camera_move: str
    duration_s: float


def build_shot_plan(segment_count: int, segment_duration_s: float) -> list[ShotPlan]:
    plans: list[ShotPlan] = []
    shot_types = ["wide", "medium", "close"]
    camera_moves = ["static", "pan", "dolly"]
    for index in range(segment_count):
        plans.append(
            ShotPlan(
                shot_id=f"shot_{index:03d}",
                shot_type=shot_types[index % len(shot_types)],
                camera_move=camera_moves[index % len(camera_moves)],
                duration_s=segment_duration_s,
            )
        )
    return plans


def summarize_shots(shots: Iterable[ShotPlan]) -> dict:
    return {
        "count": sum(1 for _ in shots),
    }
