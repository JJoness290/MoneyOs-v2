from __future__ import annotations

from dataclasses import dataclass

from app.core.visuals.ai_video.beats import Beat


@dataclass(frozen=True)
class PromptPack:
    prompt: str
    negative_prompt: str


_QUALITY_TAGS = (
    "anime, cinematic lighting, depth of field, dynamic motion, ultra detailed, dramatic shadows"
)

_NEGATIVE_PROMPT = (
    "blurry, low quality, watermark, text, logo, deformed face, extra limbs, "
    "duplicate limbs, bad anatomy, flicker"
)


def beat_to_video_prompt(beat: Beat) -> PromptPack:
    shot_tags = (
        f"{beat.camera_tag} shot, {beat.emotion_tag} mood, dynamic camera movement, parallax, "
        "wind, particles, character movement"
    )
    prompt = f"{beat.visual_intent}. {shot_tags}. {_QUALITY_TAGS}"
    return PromptPack(prompt=prompt, negative_prompt=_NEGATIVE_PROMPT)
