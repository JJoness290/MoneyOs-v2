from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScenePlan:
    camera_style: str
    action_beats: int


def default_scene_plan() -> ScenePlan:
    return ScenePlan(camera_style="key_art", action_beats=6)
