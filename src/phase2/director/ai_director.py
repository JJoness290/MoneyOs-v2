from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ShotPlan:
    camera: str
    shot_type: str
    action_bias: str
    environment_bias: str

    def as_dict(self) -> dict[str, str]:
        return {
            "camera": self.camera,
            "shot_type": self.shot_type,
            "action_bias": self.action_bias,
            "environment_bias": self.environment_bias,
        }


class AiDirector:
    def __init__(self) -> None:
        self.camera_angles = [
            "eye-level static",
            "low-angle tracking",
            "high-angle dolly",
            "overhead orbit",
            "handheld push",
            "side-profile pan",
        ]
        self.shot_types = ["wide", "medium", "close"]
        self.action_biases = [
            "dynamic movement",
            "subtle gesture",
            "character turn",
            "fast pan",
            "slow reveal",
            "interaction",
        ]
        self.environment_biases = [
            "foreground depth",
            "backlit haze",
            "neon contrast",
            "sunlit shadows",
            "silhouette",
            "reflections",
        ]

    def next_shot_plan(self, shot_index: int, recent_meta: list[dict[str, Any]]) -> dict[str, str]:
        camera = self.camera_angles[(shot_index - 1) % len(self.camera_angles)]
        shot_type = self.shot_types[(shot_index - 1) % len(self.shot_types)]
        action_bias = self.action_biases[(shot_index - 1) % len(self.action_biases)]
        environment_bias = self.environment_biases[(shot_index - 1) % len(self.environment_biases)]

        last_plan = self._last_plan(recent_meta)
        if last_plan:
            camera = self._avoid_repeat(camera, last_plan.get("camera"), self.camera_angles)
            shot_type = self._avoid_repeat(shot_type, last_plan.get("shot_type"), self.shot_types)
            action_bias = self._avoid_repeat(action_bias, last_plan.get("action_bias"), self.action_biases)
            environment_bias = self._avoid_repeat(
                environment_bias,
                last_plan.get("environment_bias"),
                self.environment_biases,
            )

        return ShotPlan(
            camera=camera,
            shot_type=shot_type,
            action_bias=action_bias,
            environment_bias=environment_bias,
        ).as_dict()

    def mutate_plan(
        self,
        plan: dict[str, str],
        min_mutations: int = 2,
        force_extreme: bool = False,
    ) -> dict[str, str]:
        mutated = dict(plan)
        mutations = []

        def rotate(value: str, options: list[str]) -> str:
            if value not in options:
                return options[0]
            idx = (options.index(value) + 1) % len(options)
            return options[idx]

        for key, options in [
            ("camera", self.camera_angles),
            ("shot_type", self.shot_types),
            ("action_bias", self.action_biases),
            ("environment_bias", self.environment_biases),
        ]:
            if force_extreme or len(mutations) < min_mutations:
                mutated[key] = rotate(mutated.get(key, ""), options)
                mutations.append(key)

        if force_extreme:
            for key, options in [
                ("camera", self.camera_angles),
                ("shot_type", self.shot_types),
                ("action_bias", self.action_biases),
                ("environment_bias", self.environment_biases),
            ]:
                mutated[key] = rotate(mutated.get(key, ""), options)

        return mutated

    def _last_plan(self, recent_meta: list[dict[str, Any]]) -> dict[str, str] | None:
        if not recent_meta:
            return None
        return recent_meta[-1].get("shot_plan")

    @staticmethod
    def _avoid_repeat(value: str, last_value: str | None, options: list[str]) -> str:
        if not last_value or value != last_value:
            return value
        if value not in options:
            return options[0]
        idx = (options.index(value) + 1) % len(options)
        return options[idx]
