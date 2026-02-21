from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
import urllib.request

USER_AGENT = "MoneyOS-CC0-Downloader/1.0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slugify(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "-" for ch in value.lower())
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-") or "download"


def _write_sources(cache_root: Path, payload: dict) -> None:
    sources_path = cache_root / "downloads" / "SOURCES.json"
    sources_path.parent.mkdir(parents=True, exist_ok=True)
    sources_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def download_url(
    url: str,
    dest_dir: Path,
    timeout: int = 60,
    retries: int = 3,
    backoff: float = 2.0,
) -> tuple[Path, str]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    slug = _slugify(url.split("//")[-1].split("/")[0])
    temp_path = dest_dir / f"{slug}_download.tmp"
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
                data = response.read()
                content_length = response.headers.get("Content-Length")
            if content_length is not None and int(content_length) != len(data):
                raise RuntimeError("download size mismatch")
            temp_path.write_bytes(data)
            sha256 = _sha256(temp_path)
            cached_dir = dest_dir / slug / sha256
            cached_dir.mkdir(parents=True, exist_ok=True)
            cached_path = cached_dir / "payload.zip"
            if cached_path.exists():
                temp_path.unlink(missing_ok=True)
                return cached_path, sha256
            temp_path.replace(cached_path)
            return cached_path, sha256
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"[DOWNLOAD] pack_id=cc0 stage=download_url source=cc0 url={url} attempt={attempt + 1} error={exc}", flush=True)
            if attempt < retries:
                time.sleep(backoff * (attempt + 1))
    raise RuntimeError(f"download failed for {url}: {last_error}")


def update_sources_manifest(
    cache_root: Path,
    source_url: str,
    license_label: str,
    license_url: str,
    sha256: str,
    cached_path: Path,
) -> None:
    sources_path = cache_root / "downloads" / "SOURCES.json"
    payload = {"sources": []}
    if sources_path.exists():
        try:
            payload = json.loads(sources_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {"sources": []}
    payload.setdefault("sources", [])
    payload["sources"].append(
        {
            "source_url": source_url,
            "license": license_label,
            "license_url": license_url,
            "sha256": sha256,
            "cached_path": str(cached_path),
        }
    )
    _write_sources(cache_root, payload)
