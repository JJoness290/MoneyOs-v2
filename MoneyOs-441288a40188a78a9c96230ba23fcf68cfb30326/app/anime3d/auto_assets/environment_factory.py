from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EnvironmentSpec:
    name: str
    vibe: str
    skyline_density: int


def default_environment() -> EnvironmentSpec:
    return EnvironmentSpec(name="city_ruins", vibe="disaster_city", skyline_density=12)


def assets_root(output_dir: Path) -> Path:
    return output_dir / "assets" / "environment"
