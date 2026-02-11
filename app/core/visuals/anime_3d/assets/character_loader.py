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
        raise RuntimeError("No anime characters are available after provisioning")
    return assets


def pick_character(seed: int, characters: list[CharacterAsset]) -> CharacterAsset:
    if not characters:
        raise RuntimeError("No characters available for selection")
    rng = random.Random(seed)
    return rng.choice(characters)
