from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScoreResult:
    score: int
    breakdown: dict[str, int]
    reason: str


def score_character(has_armature: bool, has_visemes: bool, texture_count: int) -> ScoreResult:
    if not has_armature:
        return ScoreResult(0, {"armature": 0}, "missing armature")
    score = 50
    breakdown = {"armature": 40}
    if has_visemes:
        score += 20
        breakdown["visemes"] = 20
    if texture_count > 0:
        score += 10
        breakdown["textures"] = 10
    score = min(score + 10, 100)
    breakdown["quality"] = score - sum(breakdown.values())
    return ScoreResult(score=score, breakdown=breakdown, reason="ok")


def score_environment(detail_score: int, texture_score: int) -> ScoreResult:
    score = min(100, max(0, detail_score + texture_score))
    breakdown = {"detail": detail_score, "textures": texture_score}
    return ScoreResult(score=score, breakdown=breakdown, reason="ok")


def score_audio(license_ok: bool, sample_rate_ok: bool) -> ScoreResult:
    score = 70 if license_ok else 0
    breakdown = {"license": 40 if license_ok else 0, "sample_rate": 30 if sample_rate_ok else 0}
    score = min(100, sum(breakdown.values()))
    return ScoreResult(score=score, breakdown=breakdown, reason="ok" if score >= 70 else "low")
