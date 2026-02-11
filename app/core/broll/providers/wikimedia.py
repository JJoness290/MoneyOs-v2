from __future__ import annotations

import requests

from app.core.broll.types import VideoItem
from app.core.broll.providers.base import BrollProvider


class WikimediaProvider(BrollProvider):
    def search(self, query: str, orientation: str, per_page: int) -> list[VideoItem]:
        params = {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": f"{query} filetype:video",
            "gsrlimit": per_page,
            "prop": "imageinfo",
            "iiprop": "url|size|mime|extmetadata",
        }
        print(f"[BROLL] Wikimedia search params={params}")
        response = requests.get("https://commons.wikimedia.org/w/api.php", params=params, timeout=20)
        response.raise_for_status()
        data = response.json()
        pages = data.get("query", {}).get("pages", {}).values()
        items: list[VideoItem] = []
        for page in pages:
            info = (page.get("imageinfo") or [{}])[0]
            url = info.get("url")
            mime = info.get("mime", "")
            if not url or not mime.startswith("video/"):
                continue
            width = int(info.get("width") or 0)
            height = int(info.get("height") or 0)
            ext = info.get("extmetadata") or {}
            license_name = ext.get("LicenseShortName", {}).get("value")
            license_url = ext.get("LicenseUrl", {}).get("value")
            items.append(
                VideoItem(
                    source="wikimedia",
                    provider_id=str(page.get("pageid")),
                    page_url=str(page.get("canonicalurl") or page.get("title")),
                    download_url=str(url),
                    width=width,
                    height=height,
                    duration=float(info.get("duration") or 0.0),
                    tags=[query],
                    thumbnail_url=info.get("thumburl"),
                    license=license_name,
                    license_url=license_url,
                )
            )
        return items
