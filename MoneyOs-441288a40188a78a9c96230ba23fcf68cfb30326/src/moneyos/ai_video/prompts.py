from __future__ import annotations

from src.moneyos.ai_video.beats import Beat


_STYLE_KEYWORDS = (
    "anime, cinematic lighting, depth of field, dynamic motion, ultra detailed, dramatic shadows"
)

_NEGATIVE_PROMPT = "blurry, low quality, watermark, text, deformed face, duplicate limbs"


def beat_to_video_prompt(beat: Beat) -> str:
    motion_tags = "camera pan, hair movement, particles, cinematic motion"
    return f"{beat.intent}. {motion_tags}. {_STYLE_KEYWORDS}"


def negative_prompt() -> str:
    return _NEGATIVE_PROMPT
