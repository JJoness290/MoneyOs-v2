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
    "masterpiece, best quality, cinematic anime key visual, promotional key art, "
    "dynamic perspective, dramatic rim lighting, volumetric lighting, "
    "explosive energy effects, sharp focus, detailed shading, clean lineart, "
    "depth of field, subtle film grain, 16:9"
)

_BASE_NEGATIVE = (
    "worst quality, low quality, blurry, flat lighting, bad anatomy, extra fingers, "
    "missing fingers, deformed hands, watermark, text, logo, 3d render, "
    "photorealistic, chibi"
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


def _split_identity(text: str) -> tuple[str, str]:
    parts = [part.strip() for part in text.split(".", 1)]
    if len(parts) == 2 and parts[0]:
        return parts[0], parts[1]
    return text.strip(), ""


def _limit_prompt(prompt: str, identity: str, max_words: int = 70) -> tuple[str, bool]:
    words = [word for word in prompt.split() if word.strip()]
    if len(words) <= max_words:
        return prompt, False
    tail_parts = prompt.split(",")
    keep = []
    removed = False
    for part in tail_parts:
        if identity and identity in part:
            keep.append(part)
        elif len(" ".join(keep).split()) < max_words:
            keep.append(part)
        else:
            removed = True
    trimmed = ", ".join([part.strip() for part in keep if part.strip()])
    return trimmed, removed


def build_prompt(text: str) -> PromptPayload:
    style_descriptor = _STYLE_DESCRIPTORS.get(ANIME_STYLE, _STYLE_DESCRIPTORS["thriller"])
    scene = _find_scene(text)
    identity, scene_action = _split_identity(text)
    prefix = os.getenv("MONEYOS_STYLE_PROMPT_PREFIX", "").strip()
    suffix = os.getenv("MONEYOS_STYLE_PROMPT_SUFFIX", "").strip()
    prompt = f"{_BASE_STYLE}, {identity}"
    if scene_action:
        prompt = f"{prompt}, {scene_action}"
    if scene:
        prompt = f"{prompt}, {scene}"
    prompt = f"{prompt}, {style_descriptor}"
    if prefix:
        prompt = f"{prefix}, {prompt}"
    if suffix:
        prompt = f"{prompt}, {suffix}"
    prompt, trimmed = _limit_prompt(prompt, identity)
    print(f"[ANIME_PROMPT] words={len(prompt.split())} trimmed={trimmed}")
    negative_prompt = _BASE_NEGATIVE
    return PromptPayload(prompt=prompt, negative_prompt=negative_prompt)
