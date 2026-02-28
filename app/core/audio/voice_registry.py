from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class VoiceInfo:
    id: str
    gender: str
    accent: str
    vibe: str
    recommended_use: str


VOICES = [
    VoiceInfo("anime_en_male_01", "male", "neutral", "heroic", "protagonist"),
    VoiceInfo("anime_en_male_03", "male", "neutral", "cold", "villain"),
    VoiceInfo("anime_en_female_02", "female", "neutral", "warm", "support"),
    VoiceInfo("anime_en_female_04", "female", "neutral", "energetic", "lead"),
]


class VoiceRegistry:
    def list_voices(self) -> list[dict]:
        return [v.__dict__ for v in VOICES]

    def pick_voice(self, character_name: str, mood: str, language: str = "en") -> str:
        if language.lower() != "en":
            return VOICES[0].id
        key = f"{character_name}|{mood}".encode("utf-8")
        idx = int(hashlib.sha256(key).hexdigest(), 16) % len(VOICES)
        return VOICES[idx].id
