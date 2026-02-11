from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BackendResult:
    fps: int
    resolution: str
    device: str


class BackendUnavailable(RuntimeError):
    pass


class AiVideoBackend:
    name: str

    def is_available(self) -> bool:
        raise NotImplementedError

    def load(self) -> None:
        raise NotImplementedError

    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        seconds: int,
        fps: int,
        width: int,
        height: int,
        steps: int,
        guidance: float,
        out_path: Path,
    ) -> BackendResult:
        raise NotImplementedError
