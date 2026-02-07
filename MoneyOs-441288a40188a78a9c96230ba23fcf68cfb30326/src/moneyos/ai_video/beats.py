from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
from typing import List


@dataclass(frozen=True)
class Beat:
    index: int
    timestamp_start: float
    timestamp_end: float
    intent: str
    emotion: str


_INTENT_TEMPLATES = [
    "Close-up anime character, {emotion} expression, cinematic lighting, dramatic shadows",
    "Wide shot of futuristic city, {emotion} mood, sweeping camera pan, glowing neon",
    "Action pose, {emotion} intensity, dynamic motion blur, particles flying",
    "Silhouette against sunset, {emotion} atmosphere, slow push-in, depth of field",
    "Rain-soaked street, {emotion} tension, reflections on pavement, handheld motion",
    "Studio spotlight scene, {emotion} focus, shallow depth of field, lens flare",
    "Forest clearing, {emotion} calm, drifting leaves, cinematic haze",
    "Rooftop vista, {emotion} resolve, wind in hair, orbiting camera",
    "Underground tunnel, {emotion} suspense, strobing lights, fast dolly",
    "Night sky backdrop, {emotion} wonder, floating particles, slow tilt up",
]

_EMOTIONS = [
    "determined",
    "melancholic",
    "energized",
    "tense",
    "hopeful",
    "mysterious",
    "focused",
    "intense",
    "dreamlike",
    "resolute",
]


def _seed_from_script(script: str, total_seconds: int, clip_seconds: int) -> int:
    payload = f"{script}|{total_seconds}|{clip_seconds}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:8], 16)


def generate_beats(script: str, total_seconds: int, clip_seconds: int) -> List[Beat]:
    if clip_seconds <= 0:
        raise ValueError("clip_seconds must be > 0")
    beat_count = int(total_seconds / clip_seconds)
    if beat_count <= 0:
        raise ValueError("total_seconds must be >= clip_seconds")
    rng = random.Random(_seed_from_script(script, total_seconds, clip_seconds))
    beats: list[Beat] = []
    for index in range(beat_count):
        start = index * clip_seconds
        end = start + clip_seconds
        template = _INTENT_TEMPLATES[index % len(_INTENT_TEMPLATES)]
        emotion = _EMOTIONS[index % len(_EMOTIONS)]
        if rng.random() > 0.6:
            emotion = rng.choice(_EMOTIONS)
        intent = template.format(emotion=emotion)
        beats.append(
            Beat(
                index=index,
                timestamp_start=float(start),
                timestamp_end=float(end),
                intent=intent,
                emotion=emotion,
            )
        )
    return beats
