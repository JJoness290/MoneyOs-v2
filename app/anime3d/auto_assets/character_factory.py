from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CharacterSpec:
    name: str
    palette: tuple[float, float, float]
    silhouette: str


def build_character_specs() -> list[CharacterSpec]:
    return [
        CharacterSpec(name="hero", palette=(0.8, 0.4, 0.2), silhouette="confident"),
        CharacterSpec(name="rival", palette=(0.2, 0.3, 0.7), silhouette="aggressive"),
    ]


def assets_root(output_dir: Path) -> Path:
    return output_dir / "assets" / "characters"
