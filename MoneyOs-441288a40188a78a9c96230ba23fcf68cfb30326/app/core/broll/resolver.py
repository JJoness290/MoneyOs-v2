from __future__ import annotations

import hashlib
import json
import os
import time
import shutil
from pathlib import Path
from typing import Iterable
import subprocess

import requests
from app.core.broll.cache import load_cache, write_cache
from app.core.broll.querygen import build_queries, detect_domain
from app.core.broll.ranker import rank_videos
from app.core.broll.types import VideoItem
from app.core.broll.providers.pexels import PexelsProvider
from app.core.visuals.normalize import normalize_clip


_RUN_USED_IDS: set[str] = set()
_DOWNLOADS_THIS_RUN = 0


def _provider_name() -> str:
    return os.getenv("MONEYOS_BROLL_PROVIDER", "pexels").strip().lower()


def _orientation() -> str:
    explicit = os.getenv("MONEYOS_BROLL_ORIENTATION")
    if explicit:
        return explicit.strip().lower()
    platform = os.getenv("MONEYOS_TARGET_PLATFORM", "tiktok").strip().lower()
    return "landscape" if platform == "youtube" else "portrait"


def _domain_keywords(script_text: str) -> set[str]:
    tokens = {token.lower() for token in script_text.split() if token.strip()}
    return {
        token
        for token in tokens
        if token
        in {
            "audit",
            "ledger",
            "bank",
            "transfer",
            "documents",
            "paperwork",
            "council",
            "investigation",
            "finance",
            "escrow",
            "contract",
        }
    }


def _strict_match(item: VideoItem, domain_keywords: set[str], query: str) -> bool:
    if not domain_keywords:
        return True
    haystack = " ".join([item.page_url, item.file_url, *item.tags]).lower()
    if any(term in haystack for term in ["wildlife", "nature", "forest", "giraffe", "safari", "ocean"]):
        return False
    matched = {word for word in domain_keywords if word in haystack}
    if len(matched) >= 1:
        return True
    query_tokens = {token.lower() for token in query.split() if token.strip()}
    return any(token in haystack for token in query_tokens)


def _segment_dir(segment_id: str) -> Path:
    root = Path("assets") / "broll" / "segments" / segment_id
    root.mkdir(parents=True, exist_ok=True)
    return root


def _generic_fallback_dir() -> Path:
    return Path("assets") / "broll" / "generic" / "fallback"


def _provider() -> PexelsProvider:
    provider = _provider_name()
    if provider == "pexels":
        return PexelsProvider()
    raise RuntimeError(f"Unsupported B-roll provider: {provider}")


def _safe_manifest_path(segment_dir: Path) -> Path:
    return segment_dir / "manifest.json"


def _load_manifest(segment_dir: Path) -> dict:
    path = _safe_manifest_path(segment_dir)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _write_manifest(segment_dir: Path, payload: dict) -> None:
    path = _safe_manifest_path(segment_dir)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _probe_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def _download_file(url: str, dest: Path) -> None:
    tmp_path = dest.with_suffix(".tmp")
    for attempt in range(3):
        try:
            with requests.get(url, stream=True, timeout=45) as response:
                response.raise_for_status()
                with tmp_path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            if tmp_path.stat().st_size == 0:
                raise RuntimeError("Downloaded file is empty.")
            tmp_path.replace(dest)
            return
        except Exception:
            tmp_path.unlink(missing_ok=True)
            if attempt == 2:
                raise


def _filter_existing(segment_dir: Path, items: Iterable[VideoItem]) -> list[VideoItem]:
    manifest = _load_manifest(segment_dir)
    existing_ids = {
        entry.get("provider_video_id") for entry in manifest.get("items", []) if entry.get("provider_video_id")
    }
    return [item for item in items if item.provider_id not in existing_ids and item.provider_id not in _RUN_USED_IDS]


def ensure_broll_pool(
    *,
    segment_id: str,
    segment_text: str,
    target_duration: float,
    script_text: str | None = None,
    status_callback=None,
) -> Path:
    provider = _provider()
    orientation = _orientation()
    per_segment = int(os.getenv("MONEYOS_BROLL_PER_SEGMENT", "3"))
    cache_hours = int(os.getenv("MONEYOS_BROLL_CACHE_HOURS", "24"))
    min_res = int(os.getenv("MONEYOS_BROLL_MIN_RES", "1080"))
    min_dur = float(os.getenv("MONEYOS_BROLL_MIN_DUR", "3.0"))
    max_dur = float(os.getenv("MONEYOS_BROLL_MAX_DUR", "20.0"))
    max_downloads = int(os.getenv("MONEYOS_BROLL_MAX_DOWNLOADS_PER_RUN", "60"))
    rate_sleep = float(os.getenv("MONEYOS_BROLL_RATE_SLEEP", "0.3"))
    max_attempts = int(os.getenv("MONEYOS_BROLL_MAX_ATTEMPTS_PER_SEGMENT", "8"))
    strict_mode = os.getenv("MONEYOS_BROLL_STRICT", "0") == "1"
    segment_dir = _segment_dir(segment_id)

    if status_callback:
        status_callback(
            f"[BROLL] provider={_provider_name()} orientation={orientation} "
            f"segment={segment_id} target={target_duration:.2f}s"
        )
    manifest = _load_manifest(segment_dir)
    explicit_orientation = os.getenv("MONEYOS_BROLL_ORIENTATION")
    if explicit_orientation and manifest.get("orientation") and manifest.get("orientation") != orientation:
        shutil.rmtree(segment_dir, ignore_errors=True)
        segment_dir.mkdir(parents=True, exist_ok=True)
        manifest = {}
    existing = sorted(segment_dir.glob("*.mp4"))
    if status_callback:
        status_callback(f"[BROLL] existing={len(existing)} target={per_segment}")
    if len(existing) >= per_segment:
        return segment_dir
    domain = detect_domain(script_text or segment_text)
    domain_keywords = _domain_keywords(script_text or segment_text)
    queries = build_queries(segment_text, domain)
    candidates: list[VideoItem] = []
    for query in queries:
        cached = None if cache_hours <= 0 else load_cache(_provider_name(), query, orientation, cache_hours)
        if cached:
            items = [VideoItem(**item) for item in cached.get("items", [])]
        else:
            items = provider.search(query=query, orientation=orientation, per_page=15)
            write_cache(
                _provider_name(),
                query,
                orientation,
                {"items": [item.__dict__ for item in items]},
            )
        candidates.extend(items)

    filtered = [
        item
        for item in candidates
        if item.duration >= min_dur
        and item.duration <= max_dur
        and min(item.width, item.height) >= min_res
    ]
    if strict_mode:
        filtered = [
            item for item in filtered if any(_strict_match(item, domain_keywords, query) for query in queries)
        ]
    if status_callback:
        status_callback(f"[BROLL] domain={domain} queries={queries} candidates={len(filtered)} strict={strict_mode}")

    ranked = []
    for query in queries:
        ranked.extend(
            rank_videos(
                filtered,
                query=query,
                target_duration=target_duration,
                min_res=min_res,
                orientation=orientation,
                domain=domain,
            )
        )

    ranked = _filter_existing(segment_dir, ranked)
    items = manifest.get("items", [])

    global _DOWNLOADS_THIS_RUN
    for attempt_index, item in enumerate(ranked, start=1):
        if len(items) >= per_segment:
            break
        if attempt_index > max_attempts:
            break
        if _DOWNLOADS_THIS_RUN >= max_downloads:
            break
        filename = f"clip_{len(items)+1:03d}.mp4"
        raw_path = segment_dir / f"raw_{filename}"
        dest = segment_dir / filename
        if status_callback:
            status_callback(f"[BROLL] attempt {attempt_index}/{max_attempts} -> {item.file_url}")
        try:
            _download_file(item.file_url, raw_path)
            if _probe_duration(raw_path) <= 0:
                raw_path.unlink(missing_ok=True)
                if status_callback:
                    status_callback(f"[BROLL] invalid download skipped {raw_path}")
                continue
            normalize_clip(
                raw_path,
                dest,
                duration=None,
                debug_label=None,
                status_callback=status_callback,
            )
            raw_path.unlink(missing_ok=True)
        except Exception as exc:
            if status_callback:
                status_callback(f"[BROLL] download failed {item.file_url}: {exc}")
            continue
        items.append(
            {
                "provider_video_id": item.provider_id,
                "source_page_url": item.page_url,
                "direct_file_url": item.file_url,
                "local_path": str(dest),
                "width": item.width,
                "height": item.height,
                "duration": item.duration,
            }
        )
        _RUN_USED_IDS.add(item.provider_id)
        _DOWNLOADS_THIS_RUN += 1
        if status_callback:
            status_callback(f"[BROLL] downloaded {dest}")
        time.sleep(rate_sleep)

    if status_callback:
        status_callback(f"[BROLL] segment {segment_id} pool_size={len(items)}")

    manifest_payload = {
        "segment_id": segment_id,
        "segment_text_hash": hashlib.sha1(segment_text.encode("utf-8")).hexdigest(),
        "provider": _provider_name(),
        "orientation": orientation,
        "queries": queries,
        "items": items,
    }
    _write_manifest(segment_dir, manifest_payload)
    return segment_dir


def select_broll_clip(segment_dir: Path, status_callback=None) -> Path | None:
    clips = sorted(segment_dir.glob("*.mp4"))
    if clips:
        if status_callback:
            status_callback(f"[BROLL] using clip {clips[0]}")
        return clips[0]
    generic_dir = _generic_fallback_dir()
    if generic_dir.exists():
        generic_clips = sorted(generic_dir.glob("*.mp4"))
        if generic_clips:
            if status_callback:
                status_callback(f"[BROLL] using generic fallback clip {generic_clips[0]}")
            return generic_clips[0]
    return None
