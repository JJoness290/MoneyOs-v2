from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AssetCandidate:
    asset_id: str
    name: str
    source_url: str
    license_type: str
    author: str
    download_url: str


def fetch_candidates(query: str, limit: int) -> list[AssetCandidate]:
    _ = query
    _ = limit
    return []
