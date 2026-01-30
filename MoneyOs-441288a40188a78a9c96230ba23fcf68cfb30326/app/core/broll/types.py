from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VideoItem:
    provider: str
    provider_id: str
    page_url: str
    file_url: str
    width: int
    height: int
    duration: float
    tags: list[str]
