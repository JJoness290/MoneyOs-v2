from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import shutil

from app.core.assets3d.manifest import (
    assets_by_type,
    load_manifest,
    lru_sort,
    retention_cutoff,
    write_manifest,
)
from app.core.paths import get_assets_root


@dataclass(frozen=True)
class PruneResult:
    freed_bytes: int
    removed_assets: int
    removed_temp: int


def _parse_time(value: str | None) -> datetime:
    if not value:
        return datetime(1970, 1, 1)
    return datetime.fromisoformat(value)


def _safe_unlink(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_dir():
        size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
        shutil.rmtree(path, ignore_errors=True)
        return size
    size = path.stat().st_size
    path.unlink(missing_ok=True)
    return size


def prune_assets(min_free_bytes: int, retention_days: int = 30) -> PruneResult:
    manifest = load_manifest()
    assets_root = get_assets_root()
    freed = 0
    removed_assets = 0
    removed_temp = 0
    cutoff = retention_cutoff(retention_days)

    core_keep = {
        "character": 2,
        "environment": 3,
        "sfx": 1,
        "music": 1,
    }
    keep_ids: set[str] = set()
    for asset_type, count in core_keep.items():
        records = assets_by_type(manifest, asset_type)
        top = sorted(records, key=lambda r: r.get("score_total", 0), reverse=True)[:count]
        keep_ids.update(item["asset_id"] for item in top)

    for record in lru_sort(list(manifest.get("assets", {}).values())):
        asset_id = record.get("asset_id")
        if asset_id in keep_ids:
            continue
        if record.get("pinned"):
            continue
        if record.get("in_use_by_job_ids"):
            continue
        last_used = record.get("last_used_at")
        if last_used and _parse_time(last_used) > _parse_time(cutoff):
            continue
        for local_path in record.get("local_paths", []):
            freed += _safe_unlink(Path(local_path))
        manifest["assets"].pop(asset_id, None)
        removed_assets += 1

    temp_dir = assets_root / "cache" / "downloads"
    if temp_dir.exists():
        for path in temp_dir.iterdir():
            freed += _safe_unlink(path)
            removed_temp += 1

    write_manifest(manifest)
    print(f"[PRUNE] freed={freed/1024**3:.2f}GB removed_assets={removed_assets} removed_temp={removed_temp}")
    if freed < min_free_bytes:
        return PruneResult(freed, removed_assets, removed_temp)
    return PruneResult(freed, removed_assets, removed_temp)
