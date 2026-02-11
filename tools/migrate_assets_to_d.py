from __future__ import annotations

import shutil
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source_assets = repo_root / "assets"
    target_assets = Path("D:/MoneyOS/assets")
    if not source_assets.exists():
        print("No repo assets directory found; nothing to migrate.")
        return
    if target_assets.exists() and any(target_assets.iterdir()):
        print(f"Target assets already populated: {target_assets}")
        return
    target_assets.mkdir(parents=True, exist_ok=True)
    print(f"Migrating assets from {source_assets} to {target_assets}")
    for item in source_assets.iterdir():
        dest = target_assets / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)
    print("Migration complete.")


if __name__ == "__main__":
    main()
