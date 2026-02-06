from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from app.core.visuals.anime_3d.animation_library import AnimationClip


@dataclass(frozen=True)
class RoutedAnimation:
    clip_id: str
    role: str


def route_animations(clips: Iterable[AnimationClip], roles: Iterable[str]) -> list[RoutedAnimation]:
    ordered_clips = list(clips)
    routed: list[RoutedAnimation] = []
    for index, role in enumerate(roles):
        if not ordered_clips:
            break
        clip = ordered_clips[index % len(ordered_clips)]
        routed.append(RoutedAnimation(clip_id=clip.clip_id, role=role))
    return routed
