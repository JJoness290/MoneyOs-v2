from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

ASSET_DIR_ALIASES = {
    "envs": ["envs", "environments", "environment"],
    "characters": ["characters", "chars", "character"],
    "anims": ["anims", "animations", "anim"],
    "vfx": ["vfx", "sprites", "fx"],
}


def _phase3_debug_enabled() -> bool:
    return os.getenv("MONEYOS_PHASE3_DEBUG", "0") == "1" or os.getenv("MONEYOS_DEBUG_PHASE3", "0") == "1"


def _debug_log(message: str) -> None:
    if _phase3_debug_enabled():
        print(message)


def _iter_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return (path for path in root.rglob("*") if path.is_file() and not path.name.lower().endswith(".disabled"))


def _filter_ext(paths: Iterable[Path], *extensions: str) -> list[Path]:
    allowed = {ext.lower() for ext in extensions}
    return sorted([path for path in paths if path.suffix.lower() in allowed], key=lambda p: str(p).lower())


def resolve_asset_dir(assets_dir: Path, aliases: list[str]) -> Path | None:
    for alias in aliases:
        candidate = assets_dir / alias
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def resolve_asset_dirs(assets_dir: Path) -> dict[str, Path | None]:
    return {
        "envs": resolve_asset_dir(assets_dir, ASSET_DIR_ALIASES["envs"]),
        "characters": resolve_asset_dir(assets_dir, ASSET_DIR_ALIASES["characters"]),
        "anims": resolve_asset_dir(assets_dir, ASSET_DIR_ALIASES["anims"]),
        "vfx": resolve_asset_dir(assets_dir, ASSET_DIR_ALIASES["vfx"]),
    }


def character_sort_key(candidate: Path) -> tuple[int, int, str]:
    parts = {part.lower() for part in candidate.parts}
    in_starter_pack = 0 if "starter_pack" in parts else 1
    ext_priority = 0 if candidate.suffix.lower() in {".fbx", ".glb", ".gltf", ".blend"} else 1
    return (in_starter_pack, ext_priority, candidate.name.lower())


def discover_assets(assets_dir: Path) -> tuple[dict[str, list[Path]], dict[str, Path | None]]:
    asset_dirs = resolve_asset_dirs(assets_dir)

    env_files = _filter_ext(_iter_files(asset_dirs["envs"]) if asset_dirs["envs"] else [], ".blend")
    anim_files = _filter_ext(_iter_files(asset_dirs["anims"]) if asset_dirs["anims"] else [], ".fbx")
    vfx_files = _filter_ext(_iter_files(asset_dirs["vfx"]) if asset_dirs["vfx"] else [], ".png")

    character_files: list[Path] = []
    if asset_dirs["characters"]:
        char_all = list(_iter_files(asset_dirs["characters"]))
        root_blends = [
            path
            for path in char_all
            if path.suffix.lower() == ".blend" and path.parent.resolve() == asset_dirs["characters"].resolve()
        ]
        starter_dir = asset_dirs["characters"] / "starter_pack"
        starter_blends = _filter_ext(_iter_files(starter_dir) if starter_dir.exists() else [], ".blend")
        rigged = _filter_ext(char_all, ".fbx", ".glb", ".gltf")
        character_files = sorted({*root_blends, *starter_blends, *rigged}, key=character_sort_key)

    if _phase3_debug_enabled():
        for key in ("envs", "characters", "anims", "vfx"):
            directory = asset_dirs.get(key)
            exists = bool(directory and directory.exists())
            _debug_log(f"[PHASE3_DEBUG] dir={key} path={directory} exists={exists}")
            if directory and directory.exists():
                entries = sorted([p.name for p in directory.iterdir()])[:50]
                _debug_log(f"[PHASE3_DEBUG] listdir {key}={entries}")
        _debug_log(f"[PHASE3_DEBUG] glob envs={ [p.name for p in env_files[:50]] }")
        _debug_log(f"[PHASE3_DEBUG] glob chars={ [p.name for p in character_files[:50]] }")
        _debug_log(f"[PHASE3_DEBUG] glob anims={ [p.name for p in anim_files[:50]] }")
        _debug_log(f"[PHASE3_DEBUG] glob vfx={ [p.name for p in vfx_files[:50]] }")

    return {
        "envs": env_files,
        "characters": character_files,
        "anims": anim_files,
        "vfx": vfx_files,
    }, asset_dirs


def select_env_blend(env_candidates: list[Path], environment: str) -> Path | None:
    preferred = f"{environment.strip().lower()}.blend" if environment else ""
    for candidate in env_candidates:
        if candidate.name.lower() == preferred:
            return candidate
    return env_candidates[0] if env_candidates else None


def select_character_assets(char_candidates: list[Path]) -> tuple[Path | None, Path | None]:
    ordered = sorted(char_candidates, key=character_sort_key)
    hero = next((p for p in ordered if p.name.lower() == "hero.blend"), None)
    enemy = next((p for p in ordered if p.name.lower() == "enemy.blend"), None)
    if hero is None and ordered:
        hero = ordered[0]
    if enemy is None:
        enemy = next((p for p in ordered if p != hero), None)
    return hero, enemy


def select_animation_assets(anim_candidates: list[Path]) -> dict[str, Path | None]:
    selections: dict[str, Path | None] = {"idle": None, "run": None, "punch": None}
    for candidate in anim_candidates:
        stem = candidate.stem.lower()
        if selections["idle"] is None and "idle" in stem:
            selections["idle"] = candidate
        if selections["run"] is None and ("run" in stem or "jog" in stem):
            selections["run"] = candidate
        if selections["punch"] is None and "punch" in stem:
            selections["punch"] = candidate
    return selections


def strict_assets_error(assets_dir: Path, asset_dirs: dict[str, Path | None], inventory: dict[str, list[Path]]) -> str:
    expected = [
        str((asset_dirs.get("envs") or assets_dir / "envs") / "*.blend"),
        str((asset_dirs.get("characters") or assets_dir / "characters") / "*.blend"),
        str((asset_dirs.get("characters") or assets_dir / "characters") / "starter_pack" / "*.blend"),
        str((asset_dirs.get("anims") or assets_dir / "anims") / "*.fbx"),
        str((asset_dirs.get("vfx") or assets_dir / "vfx") / "*.png"),
    ]
    found = {
        key: [p.name for p in values[:25]]
        for key, values in inventory.items()
    }
    return f"Strict asset check failed. Expected paths={expected} found={found}"
