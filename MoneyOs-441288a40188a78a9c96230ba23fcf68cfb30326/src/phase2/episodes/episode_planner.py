from __future__ import annotations

import os
import random


def plan_episode(duration_s: float) -> list[dict]:
    seed = int(os.getenv("MONEYOS_EPISODE_SEED", "0") or 0)
    rng = random.Random(seed or 42)
    beats = []
    remaining = duration_s
    shot_types = ["wide", "medium", "close"]
    locations = ["room", "street", "studio"]
    while remaining > 0.1:
        seg = min(5.0, max(2.0, rng.uniform(2.0, 4.0)))
        beats.append(
            {
                "duration": seg,
                "environment": rng.choice(locations),
                "shot": rng.choice(shot_types),
                "tag": "impact" if rng.random() > 0.7 else "dialogue",
            }
        )
        remaining -= seg
    return beats
