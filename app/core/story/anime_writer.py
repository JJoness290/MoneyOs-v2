from __future__ import annotations

import math
from typing import List

from pydantic import BaseModel, Field, ValidationError, model_validator


class Beat(BaseModel):
    beat_id: str
    duration_sec_target: float = Field(gt=0.1)
    on_screen_action: str
    dialogue: str
    emotion: str
    key_visuals: List[str] = Field(default_factory=list)
    forbidden_visuals: List[str] = Field(default_factory=list)


class Scene(BaseModel):
    scene_id: str
    location: str
    time_of_day: str
    mood: str
    stakes: str
    camera_style: str
    sfx_notes: str
    beats: List[Beat]

    @model_validator(mode="after")
    def _check_beats(self):
        if not self.beats:
            raise ValueError("scene must include at least one beat")
        return self


class Character(BaseModel):
    name: str
    archetype: str
    appearance: str
    personality: str
    voice_style: str
    catchphrase: str


class EpisodeScript(BaseModel):
    title: str
    logline: str
    themes: List[str]
    safety_notes: List[str]
    characters: List[Character]
    scenes: List[Scene]


BASE_CHARACTERS = [
    Character(
        name="Ren Aoki",
        archetype="reluctant genius",
        appearance="midnight-blue hair, silver streak, tactical academy jacket",
        personality="curious, loyal, impulsive",
        voice_style="youthful serious",
        catchphrase="We can still rewrite the future.",
    ),
    Character(
        name="Mika Sora",
        archetype="optimistic hacker",
        appearance="pink bob cut, holo visor, streetwear hoodie",
        personality="witty, warm, fearless",
        voice_style="energetic hype",
        catchphrase="Signal locked. Let's fly.",
    ),
]


def _build_beats(scene_idx: int, per_scene_seconds: float, topic_seed: str) -> list[Beat]:
    beat_count = 4
    beat_len = per_scene_seconds / beat_count
    beats: list[Beat] = []
    for i in range(beat_count):
        beats.append(
            Beat(
                beat_id=f"s{scene_idx:02d}_b{i+1:02d}",
                duration_sec_target=round(beat_len, 3),
                on_screen_action=f"Characters react to {topic_seed} escalation phase {i+1} in scene {scene_idx}.",
                dialogue=f"Narration: Scene {scene_idx} beat {i+1} pushes the conflict toward resolution.",
                emotion=("tense" if i < 2 else "hopeful"),
                key_visuals=["location continuity", "character close-up", "action emphasis"],
                forbidden_visuals=["gore", "text artifacts", "deformed faces"],
            )
        )
    return beats


def generate_anime_episode_outline_and_script(topic_seed: str, minutes: int) -> dict:
    target_minutes = max(1, int(minutes))
    target_seconds = float(target_minutes * 60)
    scene_count = max(6, int(math.ceil(target_seconds / 75.0)))
    per_scene_seconds = target_seconds / scene_count

    scenes: list[Scene] = []
    for scene_idx in range(1, scene_count + 1):
        scenes.append(
            Scene(
                scene_id=f"scene_{scene_idx:02d}",
                location=("Neo-Tokyo skybridge" if scene_idx % 2 else "Subterranean data shrine"),
                time_of_day=("night" if scene_idx % 3 else "dawn"),
                mood=("urgent" if scene_idx < scene_count else "resolved"),
                stakes="If they fail, city-wide autonomy systems collapse.",
                camera_style=("dynamic push-ins and whip pans" if scene_idx % 2 else "steady cinematic wides"),
                sfx_notes="Neon rain, distant rail hum, soft UI chirps.",
                beats=_build_beats(scene_idx, per_scene_seconds, topic_seed),
            )
        )

    payload = EpisodeScript(
        title=f"{topic_seed}: Echoes of Tomorrow",
        logline=f"In Neo-Tokyo, two young operators confront a rogue intelligence born from {topic_seed}.",
        themes=["identity", "responsibility", "hope under pressure"],
        safety_notes=["PG-13 violence only", "no self-harm", "no hate content"],
        characters=BASE_CHARACTERS,
        scenes=scenes,
    )
    try:
        validated = EpisodeScript.model_validate(payload.model_dump())
    except ValidationError as exc:
        raise RuntimeError(f"script schema validation failed: {exc}") from exc
    return validated.model_dump()
