from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Any

from app.core.paths import get_assets_root


MANIFEST_PATH = get_assets_root() / "manifest.json"


@dataclass
class AssetRecord:
    asset_id: str
    asset_type: str
    source_url: str
    license_name: str
    license_url: str
    license_proof_path: str
    size_bytes: int
    score_total: int
    score_breakdown: dict[str, int]
    pinned: bool = False
    last_used_at: str | None = None
    in_use_by_job_ids: list[str] = field(default_factory=list)
    local_paths: list[str] = field(default_factory=list)


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


def load_manifest(path: Path | None = None) -> dict[str, Any]:
    manifest_path = path or MANIFEST_PATH
    if not manifest_path.exists():
        return {"assets": {}, "updated_at": _now_iso()}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def write_manifest(payload: dict[str, Any], path: Path | None = None) -> None:
    manifest_path = path or MANIFEST_PATH
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload["updated_at"] = _now_iso()
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def upsert_asset(record: AssetRecord, path: Path | None = None) -> None:
    manifest = load_manifest(path)
    assets = manifest.setdefault("assets", {})
    assets[record.asset_id] = {
        "asset_id": record.asset_id,
        "type": record.asset_type,
        "source_url": record.source_url,
        "license_name": record.license_name,
        "license_url": record.license_url,
        "license_proof_path": record.license_proof_path,
        "size_bytes": record.size_bytes,
        "score_total": record.score_total,
        "score_breakdown": record.score_breakdown,
        "pinned": record.pinned,
        "last_used_at": record.last_used_at or _now_iso(),
        "in_use_by_job_ids": record.in_use_by_job_ids,
        "local_paths": record.local_paths,
    }
    write_manifest(manifest, path)


def update_last_used(asset_id: str, job_id: str | None = None, path: Path | None = None) -> None:
    manifest = load_manifest(path)
    assets = manifest.get("assets", {})
    record = assets.get(asset_id)
    if not record:
        return
    record["last_used_at"] = _now_iso()
    if job_id:
        in_use = set(record.get("in_use_by_job_ids", []))
        in_use.add(job_id)
        record["in_use_by_job_ids"] = sorted(in_use)
    write_manifest(manifest, path)


def clear_in_use(job_id: str, path: Path | None = None) -> None:
    manifest = load_manifest(path)
    updated = False
    for record in manifest.get("assets", {}).values():
        in_use = set(record.get("in_use_by_job_ids", []))
        if job_id in in_use:
            in_use.remove(job_id)
            record["in_use_by_job_ids"] = sorted(in_use)
            updated = True
    if updated:
        write_manifest(manifest, path)


def assets_by_type(manifest: dict[str, Any], asset_type: str) -> list[dict[str, Any]]:
    return [
        record
        for record in manifest.get("assets", {}).values()
        if record.get("type") == asset_type
    ]


def lru_sort(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(record: dict[str, Any]) -> tuple:
        last_used = record.get("last_used_at") or "1970-01-01T00:00:00"
        return (record.get("score_total", 0), last_used)

    return sorted(records, key=key)


def retention_cutoff(days: int) -> str:
    return (datetime.utcnow() - timedelta(days=days)).isoformat()
