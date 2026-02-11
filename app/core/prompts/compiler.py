from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


@dataclass(frozen=True)
class PromptCompilation:
    prompt: str
    negative_prompt: str
    trimmed: bool
    token_count: int


def _tokenize(text: str) -> list[str]:
    return [token for token in re.split(r"\s+", text.strip()) if token]


def _dedupe_phrases(phrases: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for phrase in phrases:
        cleaned = phrase.strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result


def compile_prompt(prompt: str, negative_prompt: str, max_tokens: int = 75) -> PromptCompilation:
    phrases = _dedupe_phrases(prompt.split(","))
    tokens: list[str] = []
    trimmed = False
    for phrase in phrases:
        phrase_tokens = _tokenize(phrase)
        if len(tokens) + len(phrase_tokens) <= max_tokens:
            tokens.extend(phrase_tokens)
            continue
        trimmed = True
        remaining = max_tokens - len(tokens)
        if remaining <= 0:
            break
        tokens.extend(phrase_tokens[:remaining])
        break
    compiled = " ".join(tokens)
    compiled_tokens = _tokenize(compiled)
    if len(compiled_tokens) > max_tokens:
        compiled = " ".join(compiled_tokens[:max_tokens])
        trimmed = True
    return PromptCompilation(
        prompt=compiled,
        negative_prompt=negative_prompt.strip(),
        trimmed=trimmed,
        token_count=len(_tokenize(compiled)),
    )
