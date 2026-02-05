from __future__ import annotations

import os
from pathlib import Path


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_output_root() -> Path:
    env_root = os.getenv("MONEYOS_OUTPUT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    env_short = os.getenv("MONEYOS_SHORT_WORKDIR")
    if env_short:
        return Path(env_short).expanduser().resolve()
    if os.name == "nt":
        return Path(r"C:\MoneyOS\work").resolve()
    return get_repo_root() / "output"


def get_assets_root() -> Path:
    env_assets = os.getenv("MONEYOS_ASSETS_DIR")
    if env_assets:
        return Path(env_assets).expanduser().resolve()
    env_output = os.getenv("MONEYOS_OUTPUT_ROOT")
    if env_output:
        return (Path(env_output).expanduser() / "assets").resolve()
    repo_root = get_repo_root()
    assets_dir = repo_root / "assets"
    if assets_dir.exists():
        return assets_dir.resolve()
    return repo_root.resolve()
