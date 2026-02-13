from __future__ import annotations

import json
from pathlib import Path

from app.core.assets import starter_characters as sc


def test_ensure_charpack_uses_receipt_cache(tmp_path: Path, monkeypatch) -> None:
    assets_root = tmp_path / "assets"
    char_dir = assets_root / "characters"
    pack_dir = char_dir / sc.PACK_DIR_NAME
    pack_dir.mkdir(parents=True, exist_ok=True)
    (pack_dir / "hero.fbx").write_text("x", encoding="utf-8")
    receipt = char_dir / sc.RECEIPT_NAME
    receipt.write_text(json.dumps({"ok": True, "installed": True, "source": "cache"}), encoding="utf-8")

    def _boom(_cache_zip: Path, dry_run: bool = False):
        raise AssertionError("should not download when cache is valid")

    monkeypatch.setattr(sc, "_download_zip_multi_source", _boom)
    out = sc.ensure_charpack_installed(assets_root)
    assert out == pack_dir


def test_ensure_charpack_falls_back_without_raise(tmp_path: Path, monkeypatch) -> None:
    assets_root = tmp_path / "assets"

    def _fail(_cache_zip: Path, dry_run: bool = False):
        raise RuntimeError("network down")

    monkeypatch.setattr(sc, "_download_enabled", lambda: True)
    monkeypatch.setattr(sc, "_download_zip_multi_source", _fail)

    out = sc.ensure_charpack_installed(assets_root)
    assert out.exists()
    receipt = assets_root / "characters" / sc.RECEIPT_NAME
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["installed"] is False
