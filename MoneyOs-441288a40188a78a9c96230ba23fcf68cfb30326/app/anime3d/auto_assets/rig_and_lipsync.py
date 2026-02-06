from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LipsyncPlan:
    mode: str
    has_visemes: bool


def default_lipsync() -> LipsyncPlan:
    return LipsyncPlan(mode="rms", has_visemes=False)
