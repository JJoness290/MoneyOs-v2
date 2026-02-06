from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


@dataclass(frozen=True)
class Storyboard:
    title: str
    bullets: list[str]
    lower_third: str
    evidence_label: str


_KEYWORDS = [
    "audit",
    "transfer",
    "ledger",
    "escrow",
    "bank",
    "council",
    "wire",
    "invoice",
    "deadline",
    "contract",
    "fraud",
    "balance",
    "statement",
]

_MONTHS = (
    "january february march april may june july august september october november december".split()
)


def _extract_dates(text: str) -> list[str]:
    dates = []
    for month in _MONTHS:
        match = re.search(rf"\\b{month}\\b\\s+\\d{{1,2}}(?:,\\s*\\d{{4}})?", text, re.IGNORECASE)
        if match:
            dates.append(match.group(0))
    years = re.findall(r"\\b(19\\d{2}|20\\d{2})\\b", text)
    dates.extend(years)
    return dates


def _extract_numbers(text: str) -> list[str]:
    currency = re.findall(r"(?:\\$|£|€)\\s?\\d{1,3}(?:,\\d{3})*(?:\\.\\d+)?", text)
    percents = re.findall(r"\\b\\d{1,3}%\\b", text)
    raw_numbers = re.findall(r"\\b\\d{1,3}(?:,\\d{3})+\\b", text)
    return list(dict.fromkeys(currency + percents + raw_numbers))


def _extract_entities(text: str, keywords: Iterable[str]) -> list[str]:
    found = []
    lowered = text.lower()
    for word in keywords:
        if word in lowered:
            found.append(word.upper())
    return found


def _extract_location(text: str) -> str | None:
    match = re.search(r"\\b(?:in|at)\\s+([A-Z][a-z]+(?:\\s+[A-Z][a-z]+)*)", text)
    if match:
        return match.group(1)
    return None


def _build_title(text: str) -> str:
    for keyword in _KEYWORDS:
        if keyword in text.lower():
            return f"THE {keyword.upper()}"
    words = re.findall(r"\\b[A-Z][A-Z]+\\b", text)
    if words:
        return f"THE {words[0]}"
    return "THE CASE FILE"


def _build_bullets(text: str) -> list[str]:
    bullets: list[str] = []
    numbers = _extract_numbers(text)
    if numbers:
        bullets.extend([f"FIGURE: {value}" for value in numbers[:2]])
    entities = _extract_entities(text, _KEYWORDS)
    if entities:
        bullets.extend([f"EVIDENCE: {entity}" for entity in entities[:2]])
    dates = _extract_dates(text)
    if dates:
        bullets.append(f"DATE: {dates[0].upper()}")
    if not bullets:
        tokens = re.findall(r"\\b[a-zA-Z]{6,}\\b", text.lower())
        keywords = list(dict.fromkeys(tokens))[:3]
        bullets.extend([f"KEY DETAIL: {keyword.upper()}" for keyword in keywords])
    return bullets[:4]


def extract_storyboard(segment_text: str, index: int) -> Storyboard:
    clean_text = segment_text.strip()
    title = _build_title(clean_text)
    bullets = _build_bullets(clean_text)
    location = _extract_location(clean_text)
    dates = _extract_dates(clean_text)
    if location or dates:
        left = dates[0].upper() if dates else "CASE FILE"
        right = location.upper() if location else f"SEGMENT {index:02d}"
        lower_third = f"{left} • {right}"
    else:
        lower_third = f"CASE FILE • SEGMENT {index:02d}"
    evidence_label = f"EVIDENCE LOG #{index:02d}"
    return Storyboard(title=title, bullets=bullets, lower_third=lower_third, evidence_label=evidence_label)
