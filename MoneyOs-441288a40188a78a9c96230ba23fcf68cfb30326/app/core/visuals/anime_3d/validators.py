from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ValidationReport:
    valid: bool
    message: str


def validate_render(video_path: Path) -> ValidationReport:
    if not video_path.exists():
        return ValidationReport(valid=False, message="render output missing")
    if video_path.stat().st_size == 0:
        return ValidationReport(valid=False, message="render output empty")
    return ValidationReport(valid=True, message="ok")
