from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TexturePlan:
    resolution: int
    style_preset: str
    model_path: str | None


def texture_output_dir(output_dir: Path) -> Path:
    return output_dir / "assets" / "textures"
