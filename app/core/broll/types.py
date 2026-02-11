from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VideoItem:
    source: str
    provider_id: str
    page_url: str
    download_url: str
    width: int
    height: int
    duration: float
    tags: list[str]
    thumbnail_url: str | None = None
    preview_url: str | None = None
    license: str | None = None
    license_url: str | None = None
