from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VisemeFrame:
    frame: int
    value: float


def build_jaw_frames(frame_start: int, frame_end: int, intensity: float) -> list[VisemeFrame]:
    if frame_end <= frame_start:
        return []
    return [
        VisemeFrame(frame=frame_start, value=0.0),
        VisemeFrame(frame=frame_start + 1, value=intensity),
        VisemeFrame(frame=frame_end - 1, value=intensity),
        VisemeFrame(frame=frame_end, value=0.0),
    ]
