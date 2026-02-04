from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from app.core.director.memory import DirectorMemory, load_memory, save_memory
from app.core.director.scoring import score_shot


@dataclass(frozen=True)
class DirectedShot:
    shot_id: str
    shot_type: str
    camera_move: str
    animation_clip: str
    vfx_tag: str | None


def _avoid_repeats(shot_type: str, memory: DirectorMemory) -> bool:
    return shot_type in memory.recent_shots[-2:]


def build_shot_plan(shot_candidates: Iterable[str], animation_clips: Iterable[str]) -> list[DirectedShot]:
    memory = load_memory()
    planned: list[DirectedShot] = []
    clips = list(animation_clips)
    for index, shot_type in enumerate(shot_candidates):
        if _avoid_repeats(shot_type, memory):
            shot_type = "medium"
        camera_move = "dolly" if index % 3 == 0 else "pan"
        clip = clips[index % len(clips)] if clips else "idle"
        score = score_shot(shot_type, camera_move)
        vfx_tag = "impact" if score.anime_hype > 0.6 else None
        planned.append(
            DirectedShot(
                shot_id=f"shot_{index:03d}",
                shot_type=shot_type,
                camera_move=camera_move,
                animation_clip=clip,
                vfx_tag=vfx_tag,
            )
        )
    memory = DirectorMemory(
        recent_topics=memory.recent_topics,
        recent_shots=[*memory.recent_shots, *[shot.shot_type for shot in planned]][-20:],
        recent_clips=[*memory.recent_clips, *[shot.animation_clip for shot in planned]][-50:],
    )
    save_memory(memory)
    return planned
