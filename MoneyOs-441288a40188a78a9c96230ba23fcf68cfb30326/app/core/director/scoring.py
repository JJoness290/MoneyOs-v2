from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ShotScore:
    novelty: float
    clarity: float
    action_readability: float
    dialogue_readability: float
    anime_hype: float


def score_shot(shot_type: str, camera_move: str) -> ShotScore:
    novelty = 0.7 if shot_type == "wide" else 0.6
    clarity = 0.8 if shot_type in {"medium", "close"} else 0.6
    action_readability = 0.7 if camera_move in {"dolly", "pan"} else 0.5
    dialogue_readability = 0.8 if shot_type == "close" else 0.6
    anime_hype = 0.7 if camera_move == "dolly" else 0.5
    return ShotScore(novelty, clarity, action_readability, dialogue_readability, anime_hype)
