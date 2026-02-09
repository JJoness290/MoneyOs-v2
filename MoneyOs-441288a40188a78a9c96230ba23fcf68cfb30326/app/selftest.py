from __future__ import annotations

import os
from pathlib import Path
import tempfile
import zipfile

ASSETS_ROOT = Path(tempfile.mkdtemp(prefix="moneyos_assets_"))

os.environ.setdefault("MONEYOS_TEST_MODE", "1")
os.environ.setdefault("MONEYOS_SELFTEST", "1")
os.environ.setdefault("MONEYOS_RENDER_TEMP_BUDGET_BYTES", "1")
os.environ.setdefault("MONEYOS_OUTPUT_ESTIMATE_BYTES", "1")
os.environ.setdefault("MONEYOS_MIN_FREE_BYTES", "1")
os.environ.setdefault("MONEYOS_ESTIMATED_DOWNLOAD_BYTES", "1")
os.environ.setdefault("MONEYOS_EXTRACTION_OVERHEAD_BYTES", "1")
os.environ.setdefault("MONEYOS_NORMALIZE_OVERHEAD_BYTES", "1")
os.environ.setdefault("MONEYOS_SKIP_STORAGE_CHECKS", "1")
os.environ.setdefault("MONEYOS_VISUAL_MODE", "anime_3d")
os.environ.setdefault("MONEYOS_ASSETS_DIR", str(ASSETS_ROOT))

from fastapi.testclient import TestClient

from app.core.assets3d.manifest import AssetRecord, load_manifest, upsert_asset
from app.core.assets3d.pruner import prune_assets
from app.core.assets3d.asset_pack_installer import ensure_anime3d_asset_pack, get_required_anime3d_assets
from app.core.visuals.anime_3d.blender_installer import ensure_blender_path
from app.core.visuals.anime_3d.storage import ensure_storage_budget
from app.core.paths import get_assets_root


def _test_endpoint_smoke() -> None:
    from app.main import app  # noqa: WPS433

    client = TestClient(app)
    payload = {
        "mode": "anime_auto_pro_3d",
        "duration_seconds": 60,
        "fps": 24,
        "width": 1920,
        "height": 1080,
        "seed": None,
        "enable_sfx": True,
        "enable_lipsync": True,
        "enable_music": True,
        "quality": "max",
    }
    response = client.post("/jobs/anime-episode-60s", json=payload)
    if response.status_code != 200:
        raise AssertionError(f"endpoint failed: {response.status_code} {response.text}")
    job_id = response.json().get("job_id")
    assert isinstance(job_id, str) and job_id
    status = client.get(f"/status/{job_id}")
    assert status.status_code == 200
    data = status.json()
    assert "status" in data and ("stage_key" in data or "progress_pct" in data)
    from app.main import app  # noqa: WPS433

    assert any(route.path == "/events/{job_id}" for route in app.routes)


def _test_blender_detection() -> None:
    assets_root = get_assets_root()
    fake_blender = assets_root / "tools" / "blender" / "fake_blender.exe"
    fake_blender.parent.mkdir(parents=True, exist_ok=True)
    fake_blender.write_text("fake", encoding="utf-8")
    os.environ["MONEYOS_BLENDER_PATH"] = str(fake_blender)
    path = ensure_blender_path()
    assert path.exists()
    path_file = assets_root / "tools" / "blender" / "blender_path.txt"
    path_file.write_text(str(fake_blender), encoding="utf-8")
    os.environ.pop("MONEYOS_BLENDER_PATH", None)
    path = ensure_blender_path()
    assert path.exists()


def _test_disk_space_check() -> None:
    prev = os.getenv("MONEYOS_SKIP_STORAGE_CHECKS")
    os.environ["MONEYOS_SKIP_STORAGE_CHECKS"] = "0"
    try:
        ensure_storage_budget([get_assets_root()], 10**18, "selftest")
    except RuntimeError as exc:
        assert "Insufficient disk space" in str(exc)
    else:
        raise AssertionError("disk space check did not fail")
    if prev is None:
        os.environ.pop("MONEYOS_SKIP_STORAGE_CHECKS", None)
    else:
        os.environ["MONEYOS_SKIP_STORAGE_CHECKS"] = prev


def _test_manifest_and_pruner() -> None:
    assets_root = get_assets_root()
    pinned_path = assets_root / "selftest_pinned.asset"
    in_use_path = assets_root / "selftest_inuse.asset"
    normal_path = assets_root / "selftest_normal.asset"
    for path in [pinned_path, in_use_path, normal_path]:
        path.write_text("test", encoding="utf-8")
    upsert_asset(
        AssetRecord(
            asset_id="pinned_asset",
            asset_type="character",
            source_url="local",
            license_name="CC0",
            license_url="https://creativecommons.org/publicdomain/zero/1.0/",
            license_proof_path=str(pinned_path),
            size_bytes=pinned_path.stat().st_size,
            score_total=95,
            score_breakdown={"armature": 40},
            pinned=True,
            in_use_by_job_ids=[],
            local_paths=[str(pinned_path)],
        )
    )
    upsert_asset(
        AssetRecord(
            asset_id="inuse_asset",
            asset_type="environment",
            source_url="local",
            license_name="CC0",
            license_url="https://creativecommons.org/publicdomain/zero/1.0/",
            license_proof_path=str(in_use_path),
            size_bytes=in_use_path.stat().st_size,
            score_total=90,
            score_breakdown={"detail": 40},
            pinned=False,
            in_use_by_job_ids=["job123"],
            local_paths=[str(in_use_path)],
        )
    )
    upsert_asset(
        AssetRecord(
            asset_id="normal_asset",
            asset_type="sfx",
            source_url="local",
            license_name="CC0",
            license_url="https://creativecommons.org/publicdomain/zero/1.0/",
            license_proof_path=str(normal_path),
            size_bytes=normal_path.stat().st_size,
            score_total=10,
            score_breakdown={"license": 40},
            pinned=False,
            in_use_by_job_ids=[],
            local_paths=[str(normal_path)],
        )
    )
    prune_assets(min_free_bytes=0, retention_days=0)
    manifest = load_manifest()
    assets = manifest.get("assets", {})
    assert "pinned_asset" in assets
    assert "inuse_asset" in assets


def _test_asset_pack_install() -> None:
    temp_root = Path(tempfile.mkdtemp(prefix="moneyos_pack_"))
    zip_path = temp_root / "asset_pack.zip"
    with zipfile.ZipFile(zip_path, "w") as handle:
        for rel_path in get_required_anime3d_assets():
            file_path = temp_root / rel_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text("dummy", encoding="utf-8")
            handle.write(file_path, rel_path)
    os.environ["MONEYOS_ASSET_PACK_URLS"] = zip_path.as_uri()
    ensure_anime3d_asset_pack(get_assets_root(), "selftest", strict_assets=True)
    for rel_path in get_required_anime3d_assets():
        assert (get_assets_root() / rel_path).exists()


def main() -> None:
    _test_blender_detection()
    _test_disk_space_check()
    _test_manifest_and_pruner()
    _test_asset_pack_install()
    _test_endpoint_smoke()
    print("selftest: ok")


if __name__ == "__main__":
    main()
