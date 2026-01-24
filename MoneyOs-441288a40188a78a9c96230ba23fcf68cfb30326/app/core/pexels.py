import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests

from app.config import BROLL_DIR, PEXELS_API_KEY_ENV


@dataclass
class PexelsVideo:
    video_id: int
    duration: int
    url: str


class PexelsClient:
    def __init__(self) -> None:
        api_key = os.getenv(PEXELS_API_KEY_ENV)
        if not api_key:
            raise RuntimeError("PEXELS_API_KEY is not set in the environment.")
        self._headers = {"Authorization": api_key}

    def search_videos(self, query: str, per_page: int = 8) -> list[PexelsVideo]:
        response = requests.get(
            "https://api.pexels.com/videos/search",
            headers=self._headers,
            params={"query": query, "per_page": per_page, "orientation": "portrait"},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        results = []
        for item in data.get("videos", []):
            files = item.get("video_files", [])
            best = None
            for file in files:
                if file.get("quality") == "hd" and file.get("width") and file.get("height"):
                    best = file
                    break
            if not best and files:
                best = files[0]
            if best:
                results.append(
                    PexelsVideo(
                        video_id=item.get("id"),
                        duration=item.get("duration", 0),
                        url=best.get("link"),
                    )
                )
        return results

    def download_videos(self, videos: Iterable[PexelsVideo]) -> list[Path]:
        paths: list[Path] = []
        for video in videos:
            filename = f"pexels_{video.video_id}.mp4"
            path = BROLL_DIR / filename
            if path.exists():
                paths.append(path)
                continue
            with requests.get(video.url, stream=True, timeout=60) as response:
                response.raise_for_status()
                with open(path, "wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            paths.append(path)
        return paths
