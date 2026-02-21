from __future__ import annotations

from pathlib import Path

from app.core.visuals.anime_3d.assets import character_provisioner as cp
from app.core.net.downloads import DownloadResult


def test_provision_vrm_uses_fallback_source(monkeypatch, tmp_path: Path) -> None:
    assets_root = tmp_path / "assets"
    calls: list[str] = []

    def _fake_download_from_sources(sources, dst: Path, **kwargs):  # noqa: ANN001, ANN003
        calls.extend([s.url for s in sources])
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"x" * (cp.MIN_VRM_BYTES + 8))
        return DownloadResult(ok=True, error=None, status_code=200, final_url=sources[0].url), "vrm"

    monkeypatch.setattr(cp, "download_from_sources", _fake_download_from_sources)
    records = cp.provision_anime_characters(assets_root, assets_root / "cache")
    assert records
    assert records[0].local_path.endswith("AliciaSolid_vrm-0.51.vrm")
    assert calls


def test_provision_vrm_cache_skips_download(monkeypatch, tmp_path: Path) -> None:
    assets_root = tmp_path / "assets"
    vrm_dir = assets_root / "characters" / "vrm"
    vrm_dir.mkdir(parents=True, exist_ok=True)
    cached = vrm_dir / "AliciaSolid_vrm-0.51.vrm"
    cached.write_bytes(b"x" * (cp.MIN_VRM_BYTES + 128))

    def _boom(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("download should not run for cached vrm")

    monkeypatch.setattr(cp, "download_from_sources", _boom)
    records = cp.provision_anime_characters(assets_root, assets_root / "cache")
    assert any(r.local_path.endswith("AliciaSolid_vrm-0.51.vrm") for r in records)
