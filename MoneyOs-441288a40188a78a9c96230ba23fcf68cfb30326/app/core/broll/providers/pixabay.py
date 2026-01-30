from __future__ import annotations

import os

import requests

from app.core.broll.types import VideoItem
from app.core.broll.providers.base import BrollProvider


class PixabayProvider(BrollProvider):
    def __init__(self) -> None:
        api_key = os.getenv("PIXABAY_API_KEY")
        if not api_key:
            raise RuntimeError("PIXABAY_API_KEY is required for Pixabay provider.")
        self.api_key = api_key

    def search(self, query: str, orientation: str, per_page: int) -> list[VideoItem]:
        params = {
            "key": self.api_key,
            "q": query,
            "per_page": per_page,
            "orientation": "vertical" if orientation == "portrait" else "horizontal",
        }
        print(f"[BROLL] Pixabay search params={params}")
        response = requests.get("https://pixabay.com/api/videos/", params=params, timeout=20)
        response.raise_for_status()
        data = response.json()
        items: list[VideoItem] = []
        for hit in data.get("hits", []):
            videos = hit.get("videos", {})
            best = videos.get("large") or videos.get("medium") or videos.get("small")
            if not best:
                continue
            width = int(best.get("width") or 0)
            height = int(best.get("height") or 0)
            duration = float(hit.get("duration") or 0.0)
            tags = [tag.strip().lower() for tag in str(hit.get("tags", "")).split(",") if tag.strip()]
            items.append(
                VideoItem(
                    source="pixabay",
                    provider_id=str(hit.get("id")),
                    page_url=str(hit.get("pageURL")),
                    download_url=str(best.get("url")),
                    width=width,
                    height=height,
                    duration=duration,
                    tags=tags,
                    thumbnail_url=str(hit.get("userImageURL")) if hit.get("userImageURL") else None,
                    license="Pixabay License",
                    license_url="https://pixabay.com/service/license/",
                )
            )
        return items
