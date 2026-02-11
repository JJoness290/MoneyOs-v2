from __future__ import annotations

import re
from collections import Counter

STYLE_PRESETS = {
    "moneyos_cinematic": "cinematic, high detail, moody lighting, shallow depth of field, clean composition",
    "moneyos_3d_clean": "clean 3d render, studio lighting, high detail, crisp edges, balanced composition",
    "moneyos_minimal_flat": "minimal flat illustration, clean shapes, soft gradients, balanced layout",
}

NEGATIVE_PROMPT = "text, watermark, logo, subtitles, blurry, low quality, deformed, extra limbs"

STOPWORDS = {
    "the",
    "and",
    "to",
    "of",
    "a",
    "in",
    "is",
    "it",
    "that",
    "for",
    "on",
    "with",
    "as",
    "are",
    "was",
    "be",
    "by",
    "this",
    "an",
    "or",
    "from",
    "at",
    "we",
    "you",
    "your",
    "our",
    "they",
    "their",
    "not",
}


def _extract_keywords(text: str, max_terms: int = 10) -> str:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    filtered = [token for token in tokens if token not in STOPWORDS]
    counts = Counter(filtered)
    keywords = [word for word, _ in counts.most_common(max_terms)]
    return ", ".join(keywords)


def build_segment_prompt(segment_text: str, style_preset: str) -> str:
    style = STYLE_PRESETS.get(style_preset, STYLE_PRESETS["moneyos_cinematic"])
    keywords = _extract_keywords(segment_text)
    if keywords:
        return f"{style}, {keywords}, no text, no watermark"
    return f"{style}, no text, no watermark"


def negative_prompt() -> str:
    return NEGATIVE_PROMPT
