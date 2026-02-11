from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ImportCheck:
    success: bool
    has_armature: bool
    lip_sync_ready: bool
    poly_count: int
    texture_count: int
    notes: str


def check_character_asset(asset_dir: Path) -> ImportCheck:
    rig_files = list(asset_dir.glob("**/*.fbx")) + list(asset_dir.glob("**/*.vrm"))
    has_armature = bool(rig_files)
    lip_sync_ready = bool(list(asset_dir.glob("**/*viseme*")))
    return ImportCheck(
        success=has_armature,
        has_armature=has_armature,
        lip_sync_ready=lip_sync_ready,
        poly_count=0,
        texture_count=len(list(asset_dir.glob("**/*.png"))),
        notes="placeholder import check",
    )
