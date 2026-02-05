from __future__ import annotations

from app.core.broll.types import VideoItem


def _orientation_match(item: VideoItem, orientation: str) -> float:
    portrait = item.height >= item.width
    if orientation == "portrait" and portrait:
        return 1.0
    if orientation == "landscape" and not portrait:
        return 1.0
    return 0.0


def _duration_score(item: VideoItem, target: float | None) -> float:
    if not target or target <= 0:
        return 0.0
    diff = abs(item.duration - target)
    return max(0.0, 1.0 - (diff / max(target, 1.0)))


def _resolution_score(item: VideoItem, min_res: int) -> float:
    score = min(item.width, item.height) / max(min_res, 1)
    return min(score, 2.0)


def _keyword_score(item: VideoItem, query: str) -> float:
    tokens = {token.lower() for token in query.split() if token.strip()}
    if not tokens or not item.tags:
        return 0.0
    matches = sum(1 for token in tokens if token in item.tags)
    return matches / max(len(tokens), 1)


def _domain_penalty(item: VideoItem, domain: str) -> float:
    if domain != "finance_legal":
        return 0.0
    tags = {tag.lower() for tag in item.tags}
    blacklist = {
        "wildlife",
        "nature",
        "forest",
        "mountain",
        "beach",
        "animals",
        "giraffe",
        "safari",
        "ocean",
    }
    if tags.intersection(blacklist):
        return -4.0
    return 0.0


def _domain_bonus(item: VideoItem, domain: str) -> float:
    if domain != "finance_legal":
        return 0.0
    tags = {tag.lower() for tag in item.tags}
    preferred = {
        "bank",
        "money",
        "documents",
        "paperwork",
        "audit",
        "meeting",
        "courthouse",
        "contract",
        "escrow",
        "deadline",
    }
    return 1.5 if tags.intersection(preferred) else 0.0


def rank_videos(
    candidates: list[VideoItem],
    *,
    query: str,
    target_duration: float | None,
    min_res: int,
    orientation: str,
    domain: str,
) -> list[VideoItem]:
    scored = []
    for item in candidates:
        score = (
            _duration_score(item, target_duration) * 2.0
            + _resolution_score(item, min_res) * 1.5
            + _orientation_match(item, orientation) * 2.0
            + _keyword_score(item, query) * 1.0
            + _domain_bonus(item, domain)
            + _domain_penalty(item, domain)
        )
        scored.append((score, item))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in scored]
