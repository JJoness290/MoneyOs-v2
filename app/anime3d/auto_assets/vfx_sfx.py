from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VfxPlan:
    explosion_count: int
    energy_arcs: int
    smoke_layers: int


def default_vfx() -> VfxPlan:
    return VfxPlan(explosion_count=1, energy_arcs=2, smoke_layers=3)
