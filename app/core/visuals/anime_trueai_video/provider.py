from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class ClipRequest:
    prompt: str
    negative_prompt: str
    seed: int
    seconds: int
    fps: int
    width: int
    height: int
    steps: int
    guidance: float
    out_path: Path
    target_frames: int | None = None


@dataclass(frozen=True)
class ClipResult:
    out_path: Path
    width: int
    height: int
    fps: int
    duration_s: float
    backend: str


class TextToVideoProvider(Protocol):
    name: str

    def is_available(self) -> bool:
        ...

    def generate(self, request: ClipRequest) -> ClipResult:
        ...
