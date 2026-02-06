from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PostFxPlan:
    bloom: bool
    vignette: bool
    chromatic_aberration: bool


def default_postfx() -> PostFxPlan:
    return PostFxPlan(bloom=True, vignette=True, chromatic_aberration=True)
