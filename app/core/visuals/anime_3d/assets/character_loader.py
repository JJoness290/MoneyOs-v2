from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

from app.core.visuals.anime_3d.assets.character_provisioner import (
    ProvisionedCharacter,
    load_provisioned_characters,
    provision_anime_characters,
)


@dataclass(frozen=True)
class CharacterAsset:
    name: str
    local_path: Path
    source_url: str
    license: str
    sha256: str


def _local_fallback_characters(assets_root: Path) -> list[CharacterAsset]:
    char_root = assets_root / "characters"
    candidates: list[Path] = []
    for rel in ("", "starter_pack", "kenney_animated_characters_3"):
        root = (char_root / rel) if rel else char_root
        if not root.exists():
            continue
        for pattern in ("*.blend", "*.fbx", "*.glb", "*.gltf"):
            candidates.extend(sorted(path for path in root.rglob(pattern) if path.is_file()))
    seen: set[Path] = set()
    assets: list[CharacterAsset] = []
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        assets.append(
            CharacterAsset(
                name=path.stem,
                local_path=path,
                source_url="local://starter_charpack",
                license="local",
                sha256="",
            )
        )
    return assets


def ensure_characters(assets_root: Path, cache_root: Path) -> list[CharacterAsset]:
    provisioned = load_provisioned_characters(assets_root)
    if not provisioned:
        provisioned = provision_anime_characters(assets_root, cache_root)
    assets: list[CharacterAsset] = []
    for item in provisioned:
        assets.append(
            CharacterAsset(
                name=item.name,
                local_path=Path(item.local_path),
                source_url=item.source_url,
                license=item.license,
                sha256=item.sha256,
            )
        )
    if not assets:
        assets = _local_fallback_characters(assets_root)
    return assets


def pick_character(seed: int, characters: list[CharacterAsset]) -> CharacterAsset:
    if not characters:
        raise RuntimeError("No characters available for selection")
    rng = random.Random(seed)
    return rng.choice(characters)
