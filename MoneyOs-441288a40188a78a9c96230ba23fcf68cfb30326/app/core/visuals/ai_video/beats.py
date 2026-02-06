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
    visual_intent: str
    emotion_tag: str
    camera_tag: str


_EMOTION_TAGS = ["calm", "tense", "hype", "tragic", "heroic", "mysterious", "hopeful", "determined"]
_CAMERA_TAGS = ["push-in", "pan", "orbit", "handheld", "zoom", "tilt"]

_INTENT_TEMPLATES = [
    "Close-up anime character with {emotion} expression, dramatic key light, shallow depth of field",
    "Wide cinematic cityscape at dusk, {emotion} mood, glowing neon, sweeping skyline",
    "Action cutaway with dynamic motion, {emotion} intensity, particles and motion blur",
    "Silhouette against a sunrise, {emotion} atmosphere, drifting fog, cinematic haze",
    "Rain-soaked alley, {emotion} tension, reflections on wet ground, moody lighting",
    "Studio spotlight portrait, {emotion} focus, crisp rim light, glossy highlights",
    "Forest clearing, {emotion} calm, wind-tossed leaves, ambient shafts of light",
    "Rooftop vista, {emotion} resolve, high-angle composition, skyline in distance",
    "Underground tunnel, {emotion} suspense, flickering lights, fast dolly-in",
    "Night sky scene, {emotion} wonder, floating particles, gentle tilt upward",
]


def _seed(script: str, total_seconds: int, clip_seconds: int) -> int:
    digest = hashlib.sha256(f"{script}|{total_seconds}|{clip_seconds}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def generate_beats(script: str, total_seconds: int, clip_seconds: int) -> List[Beat]:
    if clip_seconds <= 0:
        raise ValueError("clip_seconds must be > 0")
    count = int(total_seconds / clip_seconds)
    if count <= 0:
        raise ValueError("total_seconds must be >= clip_seconds")
    rng = random.Random(_seed(script, total_seconds, clip_seconds))
    beats: list[Beat] = []
    for index in range(count):
        start = index * clip_seconds
        end = start + clip_seconds
        emotion = rng.choice(_EMOTION_TAGS)
        camera = rng.choice(_CAMERA_TAGS)
        template = _INTENT_TEMPLATES[index % len(_INTENT_TEMPLATES)]
        visual = template.format(emotion=emotion)
        beats.append(
            Beat(
                index=index,
                timestamp_start=float(start),
                timestamp_end=float(end),
                visual_intent=visual,
                emotion_tag=emotion,
                camera_tag=camera,
            )
        )
    return beats
