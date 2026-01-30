from __future__ import annotations

from dataclasses import dataclass
import re

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
    prompt = (
        f"{style_descriptor}, {scene}, 16:9 composition, "
        "depth of field, atmospheric lighting, high detail, no readable text, no logos"
    )
    negative_prompt = (
        "text, watermark, signature, logo, deformed anatomy, extra limbs, lowres, blurry, "
        "bad proportions, duplicate face, readable text"
    )
    return PromptPayload(prompt=prompt, negative_prompt=negative_prompt)
