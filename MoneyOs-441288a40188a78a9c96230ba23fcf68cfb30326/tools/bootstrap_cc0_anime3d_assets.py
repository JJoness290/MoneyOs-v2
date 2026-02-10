from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.paths import get_assets_root, get_output_root
from app.core.visuals.anime_3d.blender_installer import ensure_blender_path
from src.moneyos.auto_assets.cc0_bootstrap_anime3d import ensure_cc0_anime3d_assets


def main() -> None:
    assets_root = get_assets_root()
    cache_root = get_output_root() / "auto_assets"
    blender_path = ensure_blender_path()
    report = ensure_cc0_anime3d_assets(assets_root, cache_root, blender_path, allow_network=True)
    marker_path = assets_root / ".auto_assets_installed.json"
    marker_path.write_text(
        json.dumps(
            {
                "timestamp": time.time(),
                "installed_files": report.get("installed", []),
                "sources": report.get("sources", []),
                "errors": report.get("errors", []),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
