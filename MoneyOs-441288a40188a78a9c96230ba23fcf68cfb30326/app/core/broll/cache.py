from __future__ import annotations

import hashlib
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone

from app.config import OUTPUT_DIR


def _cache_path(provider: str, query: str, orientation: str) -> Path:
    digest = hashlib.sha1(f"{provider}|{query}|{orientation}".encode("utf-8")).hexdigest()
    cache_dir = OUTPUT_DIR / "cache" / "broll" / provider
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{digest}.json"


def load_cache(provider: str, query: str, orientation: str, hours: int) -> dict | None:
    path = _cache_path(provider, query, orientation)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    created_at = payload.get("created_at")
    if not created_at:
        return None
    try:
        created_time = datetime.fromisoformat(created_at)
    except ValueError:
        return None
    if datetime.now(timezone.utc) - created_time > timedelta(hours=hours):
        return None
    return payload


def write_cache(provider: str, query: str, orientation: str, payload: dict) -> None:
    path = _cache_path(provider, query, orientation)
    payload["created_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
