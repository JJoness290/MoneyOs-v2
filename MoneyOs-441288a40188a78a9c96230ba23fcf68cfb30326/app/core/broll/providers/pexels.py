from __future__ import annotations

import os

import requests

from app.core.broll.types import VideoItem
from app.core.broll.providers.base import BrollProvider


class PexelsProvider(BrollProvider):
    def __init__(self) -> None:
        api_key = os.getenv("PEXELS_API_KEY")
        if not api_key:
            raise RuntimeError("PEXELS_API_KEY is required for Pexels provider.")
        self.api_key = api_key

    def search(self, query: str, orientation: str, per_page: int) -> list[VideoItem]:
        headers = {"Authorization": self.api_key}
        params = {"query": query, "per_page": per_page, "orientation": orientation}
        response = None
        for attempt in range(3):
            response = requests.get(
                "https://api.pexels.com/videos/search",
                headers=headers,
                params=params,
                timeout=20,
            )
            if response.status_code >= 500 and attempt < 2:
                continue
            response.raise_for_status()
            break
        if response is None:
            raise RuntimeError("Pexels request failed.")
        data = response.json()
        videos = data.get("videos", [])
        items: list[VideoItem] = []
        for video in videos:
            video_files = video.get("video_files", [])
            best_file = None
            for file in sorted(video_files, key=lambda v: v.get("width", 0) * v.get("height", 0), reverse=True):
                if file.get("file_type") != "video/mp4":
                    continue
                best_file = file
                break
            if not best_file:
                continue
            width = int(best_file.get("width") or 0)
            height = int(best_file.get("height") or 0)
            duration = float(video.get("duration") or 0.0)
            tags = [tag.lower() for tag in video.get("tags", []) if isinstance(tag, str)]
            items.append(
                VideoItem(
                    provider="pexels",
                    provider_id=str(video.get("id")),
                    page_url=str(video.get("url")),
                    file_url=str(best_file.get("link")),
                    width=width,
                    height=height,
                    duration=duration,
                    tags=tags,
                )
            )
        return items
