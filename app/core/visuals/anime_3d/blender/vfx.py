from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VfxEvent:
    frame: int
    effect: str
    intensity: float


def build_camera_shake(frame: int, intensity: float) -> VfxEvent:
    return VfxEvent(frame=frame, effect="camera_shake", intensity=intensity)
