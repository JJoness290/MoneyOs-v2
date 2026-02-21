from __future__ import annotations

import os
from pathlib import Path


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _is_windows() -> bool:
    return os.name == "nt"


def _default_roots_for_platform(is_windows: bool) -> dict[str, str]:
    if is_windows:
        return {
            "MONEYOS_ASSETS_ROOT": r"C:\MoneyOS\assets",
            "MONEYOS_OUTPUT_ROOT": r"C:\MoneyOS\work",
            "MONEYOS_CACHE_ROOT": r"C:\MoneyOS\cache",
            "HF_HOME": r"C:\MoneyOS\hf",
            "HUGGINGFACE_HUB_CACHE": r"C:\MoneyOS\hf\hub",
            "TRANSFORMERS_CACHE": r"C:\MoneyOS\hf\hub",
        }
    repo_root = get_repo_root()
    return {
        "MONEYOS_ASSETS_ROOT": str((repo_root / "assets").resolve()),
        "MONEYOS_OUTPUT_ROOT": str((repo_root / "output").resolve()),
        "MONEYOS_CACHE_ROOT": str((repo_root / "cache").resolve()),
        "HF_HOME": str((repo_root / ".cache" / "huggingface").resolve()),
        "HUGGINGFACE_HUB_CACHE": str((repo_root / ".cache" / "huggingface" / "hub").resolve()),
        "TRANSFORMERS_CACHE": str((repo_root / ".cache" / "huggingface" / "hub").resolve()),
    }


def apply_default_storage_env() -> None:
    defaults = _default_roots_for_platform(_is_windows())
    for key, value in defaults.items():
        os.environ.setdefault(key, value)
    if _is_windows():
        os.environ.setdefault("HUGGINGFACE_HUB_DISABLE_SYMLINKS", "1")
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def _resolve_dir(env_name: str, fallback: str | Path) -> Path:
    raw = os.getenv(env_name)
    base = Path(raw) if raw else Path(fallback)
    path = base.expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_output_root() -> Path:
    apply_default_storage_env()
    return _resolve_dir("MONEYOS_OUTPUT_ROOT", _default_roots_for_platform(_is_windows())["MONEYOS_OUTPUT_ROOT"])


def get_assets_root() -> Path:
    apply_default_storage_env()
    env_assets = os.getenv("MONEYOS_ASSETS_DIR")
    if env_assets:
        return _resolve_dir("MONEYOS_ASSETS_DIR", env_assets)
    return _resolve_dir("MONEYOS_ASSETS_ROOT", _default_roots_for_platform(_is_windows())["MONEYOS_ASSETS_ROOT"])


def get_cache_root() -> Path:
    apply_default_storage_env()
    return _resolve_dir("MONEYOS_CACHE_ROOT", _default_roots_for_platform(_is_windows())["MONEYOS_CACHE_ROOT"])


def get_hf_home() -> Path:
    apply_default_storage_env()
    return _resolve_dir("HF_HOME", _default_roots_for_platform(_is_windows())["HF_HOME"])


def get_hf_hub_cache() -> Path:
    apply_default_storage_env()
    return _resolve_dir("HUGGINGFACE_HUB_CACHE", _default_roots_for_platform(_is_windows())["HUGGINGFACE_HUB_CACHE"])


def get_characters_dir() -> Path:
    path = (get_assets_root() / "characters").resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


apply_default_storage_env()
