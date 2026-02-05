from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from app.config import (
    ASSET_KEEP_TOP_N,
    ASSET_LICENSE_MODE,
    ASSET_MAX_DOWNLOADS_PER_RUN,
    ASSET_PROVIDERS,
    ASSET_REVIEW_MODE,
    AUTO_CHARACTERS_DIR,
    SKETCHFAB_API_TOKEN,
)
from app.core.assets.harvester.blender_import_check import check_character_asset
from app.core.assets.harvester.cache import get_cache_paths, write_json
from app.core.assets.harvester.character_judge import score_character
from app.core.assets.harvester.license_guard import check_license
from app.core.assets.harvester.providers import opengameart, sketchfab, vroidhub


@dataclass(frozen=True)
class HarvestReport:
    candidates: list[dict]
    selected: list[dict]


def _download_placeholder(asset_id: str) -> Path:
    asset_dir = AUTO_CHARACTERS_DIR / asset_id
    asset_dir.mkdir(parents=True, exist_ok=True)
    (asset_dir / "meta.json").write_text("{}", encoding="utf-8")
    return asset_dir


def _gather_candidates(query: str, count: int) -> list[dict]:
    candidates: list[dict] = []
    remaining = count
    if "opengameart" in ASSET_PROVIDERS:
        for candidate in opengameart.fetch_candidates(query, remaining):
            candidates.append(candidate.__dict__)
        remaining = max(0, count - len(candidates))
    if "sketchfab" in ASSET_PROVIDERS:
        for candidate in sketchfab.fetch_candidates(query, remaining, SKETCHFAB_API_TOKEN):
            candidates.append(candidate.__dict__)
        remaining = max(0, count - len(candidates))
    if "vroidhub" in ASSET_PROVIDERS:
        for candidate in vroidhub.fetch_candidates(query, remaining):
            candidates.append(candidate.__dict__)
    return candidates[:count]


def _score_candidates(candidates: Iterable[dict]) -> list[dict]:
    scored: list[dict] = []
    for candidate in candidates:
        license_result = check_license(candidate.get("license_type"))
        if not license_result.allowed:
            candidate["license_status"] = license_result.reason
            candidate["score"] = 0
            candidate["tags"] = ["license_rejected"]
            scored.append(candidate)
            continue
        asset_dir = _download_placeholder(candidate["asset_id"])
        check = check_character_asset(asset_dir)
        score = score_character(check)
        candidate.update(
            {
                "license_status": license_result.reason,
                "score": score.score,
                "tags": score.tags,
                "review_notes": score.reason,
                "asset_dir": str(asset_dir),
            }
        )
        scored.append(candidate)
    scored.sort(key=lambda item: item.get("score", 0), reverse=True)
    return scored


def _select_top(scored: list[dict]) -> list[dict]:
    if ASSET_REVIEW_MODE:
        return []
    return scored[:ASSET_KEEP_TOP_N]


def _write_credits(selected: Iterable[dict]) -> None:
    paths = get_cache_paths()
    credits_payload = []
    credits_lines = []
    for item in selected:
        credits_payload.append(
            {
                "asset_id": item.get("asset_id"),
                "name": item.get("name"),
                "author": item.get("author"),
                "license": item.get("license_type"),
                "source_url": item.get("source_url"),
            }
        )
        credits_lines.append(
            f"{item.get('name')} by {item.get('author')} ({item.get('license_type')}): {item.get('source_url')}"
        )
    write_json(paths.credits_path, credits_payload)
    paths.credits_text_path.write_text("\n".join(credits_lines), encoding="utf-8")


def harvest_assets(query: str, style: str, count: int) -> HarvestReport:
    _ = style
    count = min(count, ASSET_MAX_DOWNLOADS_PER_RUN)
    candidates = _gather_candidates(query, count)
    scored = _score_candidates(candidates)
    selected = _select_top(scored)
    paths = get_cache_paths()
    write_json(paths.report_path, {"license_mode": ASSET_LICENSE_MODE, "candidates": scored})
    write_json(paths.selected_path, {"selected": selected})
    _write_credits(selected)
    return HarvestReport(candidates=scored, selected=selected)
