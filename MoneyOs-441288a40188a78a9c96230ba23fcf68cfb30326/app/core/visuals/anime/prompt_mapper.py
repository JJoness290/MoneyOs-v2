from __future__ import annotations

from dataclasses import dataclass
import re

import os

from app.config import ANIME_STYLE


@dataclass(frozen=True)
class PromptPayload:
    prompt: str
    negative_prompt: str


_STYLE_DESCRIPTORS = {
    "thriller": "cinematic anime illustration, thriller mood, dramatic shadows, clean linework, subtle film grain",
    "clean": "cinematic anime illustration, clean modern look, crisp linework, soft lighting, minimal noise",
    "cyberpunk": "cinematic anime illustration, neon cyberpunk palette, high-contrast lighting, futuristic tech glow",
}

_BASE_STYLE = (
    "masterpiece, best quality, ultra-detailed, high-impact anime key visual, "
    "cinematic anime illustration, anime movie style, dynamic action pose, "
    "dramatic perspective, explosive energy effects, strong rim lighting, "
    "volumetric lighting, cinematic color grading, warm highlights, cool shadows, "
    "foreground action background chaos, depth of field, clean lineart, "
    "detailed shading, promotional key art"
)

_BASE_NEGATIVE = (
    "worst quality, low quality, blurry, flat lighting, bad anatomy, extra fingers, "
    "missing fingers, deformed hands, deformed face, 3d render, photorealistic, "
    "western cartoon, chibi, watermark, text, logo"
)

_SCENE_TEMPLATES: list[tuple[str, str]] = [
    ("audit", "forensic audit room, documents spread, investigator reviewing ledgers"),
    ("ledger", "financial ledger close-up, meticulous notes, desk lamp lighting"),
    ("escrow", "corporate meeting room, escrow documents, tense negotiation vibe"),
    ("bank", "modern bank interior, transaction screens glowing, late-night operations"),
    ("court", "courtroom hallway, legal files stacked, quiet tension"),
    ("deadline", "late-night office, clock ticking, urgent paperwork"),
    ("investigation", "investigation board with pinned photos, evidence strings, focused detective"),
    ("transfer", "wire transfer confirmation on secure terminal, security focus"),
    ("contract", "contract review table, signatures pending, serious mood"),
    ("fraud", "shadowy office, flagged transactions on monitors, alert atmosphere"),
]

_KEYWORDS = [keyword for keyword, _ in _SCENE_TEMPLATES]


def _find_scene(text: str) -> str:
    lowered = text.lower()
    for keyword, scene in _SCENE_TEMPLATES:
        if keyword in lowered:
            return scene
    tokens = re.findall(r"\\b[a-zA-Z]{5,}\\b", lowered)
    if tokens:
        return f"focused analyst in office, {tokens[0]} context, investigative mood"
    return "focused analyst in office, investigative mood"


def build_prompt(text: str) -> PromptPayload:
    style_descriptor = _STYLE_DESCRIPTORS.get(ANIME_STYLE, _STYLE_DESCRIPTORS["thriller"])
    scene = _find_scene(text)
    prefix = os.getenv("MONEYOS_STYLE_PROMPT_PREFIX", "").strip()
    suffix = os.getenv("MONEYOS_STYLE_PROMPT_SUFFIX", "").strip()
    prompt = (
        f"{_BASE_STYLE}, {style_descriptor}, {scene}, 16:9 composition, "
        "depth of field, atmospheric lighting, high detail, no readable text, no logos"
    )
    if prefix:
        prompt = f"{prefix}, {prompt}"
    if suffix:
        prompt = f"{prompt}, {suffix}"
    negative_prompt = _BASE_NEGATIVE
    return PromptPayload(prompt=prompt, negative_prompt=negative_prompt)
