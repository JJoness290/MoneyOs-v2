from __future__ import annotations

from app.core.paths import get_assets_root, get_output_root
from app.core.visuals.anime_3d.render_pipeline import _required_asset_paths


def main() -> None:
    assets_root = get_assets_root()
    output_root = get_output_root()
    print(f"assets_root={assets_root}")
    print(f"output_root={output_root}")
    required = _required_asset_paths()
    missing = [key for key, path in required.items() if not path.exists()]
    if missing:
        print("Missing assets:")
        for key in missing:
            print(f"- {key}")
    else:
        print("All required assets found.")


if __name__ == "__main__":
    main()
