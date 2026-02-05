from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RenderPlan:
    resolution: tuple[int, int]
    fps: int
    seconds: int
    quality: str


def output_root(output_dir: Path) -> Path:
    return output_dir / "assets"
