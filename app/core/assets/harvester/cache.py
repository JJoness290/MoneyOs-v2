from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.config import OUTPUT_DIR


@dataclass(frozen=True)
class CachePaths:
    report_path: Path
    selected_path: Path
    credits_path: Path
    credits_text_path: Path


def get_cache_paths() -> CachePaths:
    assets_dir = OUTPUT_DIR / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    return CachePaths(
        report_path=assets_dir / "harvest_report.json",
        selected_path=assets_dir / "selected_characters.json",
        credits_path=assets_dir / "credits_assets.json",
        credits_text_path=assets_dir / "CREDITS.txt",
    )


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
