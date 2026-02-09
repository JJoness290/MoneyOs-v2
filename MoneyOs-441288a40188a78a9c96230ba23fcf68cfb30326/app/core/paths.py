from __future__ import annotations

import os
from pathlib import Path


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_output_root() -> Path:
    env_root = os.getenv("MONEYOS_OUTPUT_ROOT")
    if env_root:
        path = Path(env_root).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path
    env_short = os.getenv("MONEYOS_SHORT_WORKDIR")
    if env_short:
        path = Path(env_short).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path
    drive_d = Path("D:/")
    if drive_d.exists():
        path = Path("D:/MoneyOS/work").resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path
    if os.name == "nt":
        path = Path(r"C:\MoneyOS\work").resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path
    path = (get_repo_root() / "output").resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_assets_root() -> Path:
    env_assets_root = os.getenv("MONEYOS_ASSETS_ROOT")
    if env_assets_root:
        path = Path(env_assets_root).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path
    env_assets = os.getenv("MONEYOS_ASSETS_DIR")
    if env_assets:
        path = Path(env_assets).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path
    drive_d = Path("D:/")
    if drive_d.exists():
        path = Path("D:/MoneyOS/assets").resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path
    env_output = os.getenv("MONEYOS_OUTPUT_ROOT")
    if env_output:
        path = (Path(env_output).expanduser() / "assets").resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path
    repo_root = get_repo_root()
    assets_dir = repo_root / "assets"
    if assets_dir.exists():
        path = assets_dir.resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path
    path = repo_root.resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path
