from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.config import CHARACTERS_DIR, OUTPUT_DIR
from app.core.visuals.anime_3d.blender_runner import BlenderCommand, run_blender


@dataclass(frozen=True)
class RetargetResult:
    character_id: str
    clip_id: str
    output_path: Path
    success: bool
    error: str | None = None


def _load_bone_map(character_dir: Path) -> dict:
    meta_path = character_dir / "meta.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8")).get("bone_map", {})
    except json.JSONDecodeError:
        return {}


def retarget_clip(character_id: str, clip_path: Path) -> RetargetResult:
    character_dir = CHARACTERS_DIR / character_id
    output_dir = OUTPUT_DIR / "retargeted" / character_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{clip_path.stem}.blend"
    bone_map = _load_bone_map(character_dir)
    script_path = Path(__file__).parent / "blender" / "retarget.py"
    try:
        run_blender(
            BlenderCommand(
                script_path=script_path,
                args=[
                    "--character",
                    str(character_dir),
                    "--clip",
                    str(clip_path),
                    "--output",
                    str(output_path),
                    "--bone-map",
                    json.dumps(bone_map),
                ],
            )
        )
        return RetargetResult(character_id=character_id, clip_id=clip_path.stem, output_path=output_path, success=True)
    except Exception as exc:  # noqa: BLE001
        return RetargetResult(
            character_id=character_id,
            clip_id=clip_path.stem,
            output_path=output_path,
            success=False,
            error=str(exc),
        )
