from __future__ import annotations

import os
from pathlib import Path
import tempfile

import pytest

from src.moneyos.auto_assets.cc0_bootstrap_anime3d import (
    extract_oga_zip_url,
    verify_cc0_in_html,
    ensure_cc0_anime3d_assets,
)


def test_verify_cc0_in_html() -> None:
    html = "<html><body>License: CC0 <a href='https://creativecommons.org/publicdomain/zero/1.0/'>CC0</a></body></html>"
    assert verify_cc0_in_html(html)


def test_extract_oga_zip_url() -> None:
    html = (
        "<a href='/sites/default/files/2024-01/animated-human.zip'>Download</a>"
        "<a href='https://example.com/other.zip'>Other</a>"
    )
    assert extract_oga_zip_url(html) == "/sites/default/files/2024-01/animated-human.zip"


@pytest.mark.skipif(os.getenv("MONEYOS_TEST_REAL_NET") != "1", reason="real network test")
def test_real_cc0_bootstrap() -> None:
    assets_root = Path(tempfile.mkdtemp(prefix="moneyos_cc0_assets_"))
    cache_root = Path(tempfile.mkdtemp(prefix="moneyos_cc0_cache_"))
    report = ensure_cc0_anime3d_assets(assets_root, cache_root, blender_path=None, allow_network=True)
    assert report["installed"]
    required = [
        "characters/hero.blend",
        "characters/enemy.blend",
        "envs/city.blend",
        "anims/idle.fbx",
        "anims/run.fbx",
        "anims/punch.fbx",
    ]
    for rel in required:
        assert (assets_root / rel).exists()
