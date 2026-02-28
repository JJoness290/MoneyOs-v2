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


def _emotion_for_progress(progress: float) -> str:
    if progress < 0.3:
        return "anxious"
    if progress < 0.7:
        return "urgent"
    if progress < 0.95:
        return "desperate"
    return "relieved"


def _ren_line(topic_seed: str, progress: float) -> str:
    if progress < 0.3:
        return f"Mika, the {topic_seed.lower()} spike just tore through district relays—if we stall, the whole grid falls out of sync."
    if progress < 0.7:
        return "I can hold the breach for thirty seconds, but after that the city core will lock us out for good."
    if progress < 0.95:
        return "I can feel the reactor shaking under us—if this fails, everyone above this shrine loses power and life support."
    return "We did it… the signal is stabilizing. Neo-Tokyo gets to see the sunrise after all."


def _mika_line(topic_seed: str, progress: float) -> str:
    if progress < 0.3:
        return f"Then we move now, Ren. I’m rerouting the {topic_seed.lower()} pulse through the shrine node before it eats the transit net."
    if progress < 0.7:
        return "Keep talking to me—your timing is my metronome. If we break rhythm once, the cascade wins."
    if progress < 0.95:
        return "Don’t you dare let go. I’ve got the kill-switch mapped, and when I count down we hit it together."
    return "You gave me the window I needed. Core seal is clean, and the city lights are coming back one by one."


def _build_beats(scene_idx: int, scene_count: int, per_scene_seconds: float, topic_seed: str) -> list[Beat]:
    beat_count = 4
    beat_len = per_scene_seconds / beat_count
    beats: list[Beat] = []
    for beat_idx in range(beat_count):
        progress = ((scene_idx - 1) * beat_count + beat_idx + 1) / (scene_count * beat_count)
        speaker = "Ren Aoki" if beat_idx % 2 == 0 else "Mika Sora"
        spoken = _ren_line(topic_seed, progress) if speaker == "Ren Aoki" else _mika_line(topic_seed, progress)
        beats.append(
            Beat(
                beat_id=f"s{scene_idx:02d}_b{beat_idx+1:02d}",
                duration_sec_target=round(beat_len, 3),
                on_screen_action=(
                    "Rain-slick neon reflections ripple across armored rails while camera glides past sparking conduits, "
                    "framing both leads as alarms pulse and distant towers flicker."
                ),
                dialogue=f"{speaker}: {spoken}",
                emotion=_emotion_for_progress(progress),
                key_visuals=["neon storm skyline", "reactor light pulses", "close emotional framing"],
                forbidden_visuals=["gore", "text overlays", "deformed anatomy"],
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
        progress = scene_idx / scene_count
        scenes.append(
            Scene(
                scene_id=f"scene_{scene_idx:02d}",
                location=("Neo-Tokyo skybridge" if scene_idx % 2 else "Subterranean data shrine"),
                time_of_day=("night" if scene_idx < scene_count else "dawn"),
                mood=("volatile" if progress < 0.7 else "high-stakes resolve" if progress < 0.95 else "release"),
                stakes="If they fail, city-wide autonomy systems collapse.",
                camera_style=("dynamic push-ins and whip pans" if scene_idx % 2 else "steady cinematic wides with hard light contrast"),
                sfx_notes="Neon rain, distant rail hum, capacitor whine, reactor tremors.",
                beats=_build_beats(scene_idx, scene_count, per_scene_seconds, topic_seed),
            )
        )

    payload = EpisodeScript(
        title=f"{topic_seed}: Echoes of Tomorrow",
        logline=f"In Neo-Tokyo, Ren and Mika race through a collapsing network to stop the intelligence born from {topic_seed}.",
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
