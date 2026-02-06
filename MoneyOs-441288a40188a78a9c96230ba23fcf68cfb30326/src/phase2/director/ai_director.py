from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


CAMERA_ANGLES = [
    "low-angle tracking",
    "high-angle static",
    "eye-level handheld",
    "overhead top-down",
    "shoulder-level pan",
    "dutch tilt",
    "crane rise",
]

SHOT_TYPES = [
    "wide",
    "medium",
    "close",
    "extreme close",
    "establishing",
]

ACTION_BIASES = [
    "dynamic movement",
    "subtle gesture",
    "rapid action",
    "slow reveal",
    "impact moment",
]

ENVIRONMENT_BIASES = [
    "foreground depth",
    "backlit atmosphere",
    "reflective surfaces",
    "textured background",
    "open space",
]


@dataclass
class ShotPlan:
    camera: str
    shot_type: str
    action_bias: str
    environment_bias: str


def _recent_values(recent_meta: Iterable[dict], key: str) -> set[str]:
    values = set()
    for item in recent_meta:
        value = item.get(key)
        if value:
            values.add(str(value))
    return values


def _rotate_choice(options: list[str], index: int, avoid: set[str]) -> str:
    if not options:
        return ""
    for offset in range(len(options)):
        candidate = options[(index + offset) % len(options)]
        if candidate not in avoid:
            return candidate
    return options[index % len(options)]


def next_shot_plan(shot_index: int, recent_meta: Iterable[dict]) -> dict:
    recent_meta_list = list(recent_meta)
    camera = _rotate_choice(CAMERA_ANGLES, shot_index, _recent_values(recent_meta_list, "camera"))
    shot_type = _rotate_choice(SHOT_TYPES, shot_index + 1, _recent_values(recent_meta_list, "shot_type"))
    action_bias = _rotate_choice(
        ACTION_BIASES, shot_index + 2, _recent_values(recent_meta_list, "action_bias")
    )
    environment_bias = _rotate_choice(
        ENVIRONMENT_BIASES, shot_index + 3, _recent_values(recent_meta_list, "environment_bias")
    )
    return ShotPlan(
        camera=camera,
        shot_type=shot_type,
        action_bias=action_bias,
        environment_bias=environment_bias,
    ).__dict__


def mutate_shot_plan(base_plan: dict, attempt_index: int) -> dict:
    mutated = dict(base_plan)
    mutation_targets = ["camera", "shot_type", "action_bias", "environment_bias"]
    start = attempt_index % len(mutation_targets)
    mutate_keys = {mutation_targets[start], mutation_targets[(start + 1) % len(mutation_targets)]}

    if "camera" in mutate_keys:
        mutated["camera"] = CAMERA_ANGLES[(attempt_index + 2) % len(CAMERA_ANGLES)]
    if "shot_type" in mutate_keys:
        mutated["shot_type"] = SHOT_TYPES[(attempt_index + 3) % len(SHOT_TYPES)]
    if "action_bias" in mutate_keys:
        mutated["action_bias"] = ACTION_BIASES[(attempt_index + 4) % len(ACTION_BIASES)]
    if "environment_bias" in mutate_keys:
        mutated["environment_bias"] = ENVIRONMENT_BIASES[(attempt_index + 5) % len(ENVIRONMENT_BIASES)]

    return mutated


def force_extreme_mutation(base_plan: dict, shot_index: int) -> dict:
    mutated = dict(base_plan)
    mutated["camera"] = CAMERA_ANGLES[(shot_index + 5) % len(CAMERA_ANGLES)]
    mutated["shot_type"] = SHOT_TYPES[(shot_index + 4) % len(SHOT_TYPES)]
    mutated["action_bias"] = ACTION_BIASES[(shot_index + 3) % len(ACTION_BIASES)]
    mutated["environment_bias"] = ENVIRONMENT_BIASES[(shot_index + 2) % len(ENVIRONMENT_BIASES)]
    return mutated
