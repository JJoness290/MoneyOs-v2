from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import random


@dataclass(frozen=True)
class CharacterVariation:
    hair_color: tuple[float, float, float, float]
    eye_color: tuple[float, float, float, float]
    skin_tone_shift: float
    clothing_color: tuple[float, float, float, float]

    def to_json(self) -> str:
        return json.dumps(asdict(self))


_HAIR = [
    (0.12, 0.13, 0.16, 1.0),
    (0.72, 0.32, 0.22, 1.0),
    (0.16, 0.2, 0.42, 1.0),
    (0.52, 0.1, 0.25, 1.0),
]

_EYES = [
    (0.18, 0.56, 0.95, 1.0),
    (0.23, 0.72, 0.52, 1.0),
    (0.62, 0.37, 0.9, 1.0),
    (0.82, 0.45, 0.2, 1.0),
]

_CLOTHES = [
    (0.18, 0.21, 0.55, 1.0),
    (0.56, 0.12, 0.12, 1.0),
    (0.18, 0.45, 0.34, 1.0),
    (0.43, 0.27, 0.12, 1.0),
]


def build_character_variation(seed: int) -> CharacterVariation:
    rng = random.Random(seed)
    return CharacterVariation(
        hair_color=rng.choice(_HAIR),
        eye_color=rng.choice(_EYES),
        skin_tone_shift=round(rng.uniform(-0.08, 0.08), 3),
        clothing_color=rng.choice(_CLOTHES),
    )
