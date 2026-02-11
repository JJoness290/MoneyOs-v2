from __future__ import annotations

from dataclasses import dataclass

from app.core.assets.harvester.blender_import_check import ImportCheck


@dataclass(frozen=True)
class CharacterScore:
    score: int
    tags: list[str]
    reason: str


def score_character(check: ImportCheck) -> CharacterScore:
    tags: list[str] = []
    score = 50
    if not check.has_armature:
        return CharacterScore(score=0, tags=["no_rig"], reason="missing armature")
    tags.append("anime_ready")
    if check.lip_sync_ready:
        tags.append("lip_sync_ready")
        score += 20
    if check.texture_count == 0:
        tags.append("missing_textures")
        score -= 10
    return CharacterScore(score=max(score, 0), tags=tags, reason=check.notes)
