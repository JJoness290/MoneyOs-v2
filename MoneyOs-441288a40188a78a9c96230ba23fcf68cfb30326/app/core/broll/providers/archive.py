from __future__ import annotations

import requests

from app.core.broll.types import VideoItem
from app.core.broll.providers.base import BrollProvider


class ArchiveProvider(BrollProvider):
    def search(self, query: str, orientation: str, per_page: int) -> list[VideoItem]:
        params = {
            "q": f"{query} mediatype:movies",
            "fl[]": ["identifier", "title"],
            "rows": per_page,
            "output": "json",
        }
        print(f"[BROLL] Archive search params={params}")
        response = requests.get("https://archive.org/advancedsearch.php", params=params, timeout=20)
        response.raise_for_status()
        data = response.json()
        docs = data.get("response", {}).get("docs", [])
        items: list[VideoItem] = []
        for doc in docs:
            identifier = doc.get("identifier")
            if not identifier:
                continue
            page_url = f"https://archive.org/details/{identifier}"
            download_url = f"https://archive.org/download/{identifier}/{identifier}.mp4"
            items.append(
                VideoItem(
                    source="archive",
                    provider_id=identifier,
                    page_url=page_url,
                    download_url=download_url,
                    width=0,
                    height=0,
                    duration=0.0,
                    tags=[query],
                    license="Internet Archive",
                    license_url="https://archive.org/about/terms.php",
                )
            )
        return items
