from __future__ import annotations

import json
from pathlib import Path

from app.core.paths import get_assets_root, get_output_root
from app.core.visuals.anime_3d.blender_installer import ensure_blender_path
from src.moneyos.auto_assets.cc0_bootstrap_anime3d import ensure_cc0_anime3d_assets


def main() -> None:
    assets_root = get_assets_root()
    cache_root = get_output_root() / "auto_assets"
    blender_path = ensure_blender_path()
    report = ensure_cc0_anime3d_assets(assets_root, cache_root, blender_path, allow_network=True)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
