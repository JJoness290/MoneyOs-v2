from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from pathlib import Path

from src.utils.win_paths import get_short_workdir, safe_join, shorten_component
from src.phase2.assets.validators import verify_manifest_has_license


class AssetSource:
    name = "base"

    def list(self, kind: str) -> list[dict]:
        raise NotImplementedError

    def download(self, asset: dict, target_dir: Path) -> dict:
        raise NotImplementedError


class PolyHavenSource(AssetSource):
    name = "polyhaven"

    def list(self, kind: str) -> list[dict]:
        if kind != "hdri":
            return []
        return [
            {
                "id": "noon_grass",
                "url": "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/noon_grass_1k.hdr",
                "license": "CC0",
                "author": "Poly Haven",
            }
        ]

    def download(self, asset: dict, target_dir: Path) -> dict:
        target_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{asset['id']}.hdr"
        dest = target_dir / filename
        urllib.request.urlretrieve(asset["url"], dest)  # noqa: S310
        manifest = {
            "id": asset["id"],
            "source": self.name,
            "url": asset["url"],
            "license": asset["license"],
            "author": asset["author"],
            "files": [{"path": str(dest)}],
        }
        (target_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest


class OpenGameArtSource(AssetSource):
    name = "opengameart"

    def list(self, kind: str) -> list[dict]:
        if kind != "texture":
            return []
        return [
            {
                "id": "oga_checker",
                "url": "https://opengameart.org/sites/default/files/OGA-Checker.png",
                "license": "CC0",
                "author": "OpenGameArt",
            }
        ]

    def download(self, asset: dict, target_dir: Path) -> dict:
        target_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{asset['id']}.png"
        dest = target_dir / filename
        urllib.request.urlretrieve(asset["url"], dest)  # noqa: S310
        manifest = {
            "id": asset["id"],
            "source": self.name,
            "url": asset["url"],
            "license": asset["license"],
            "author": asset["author"],
            "files": [{"path": str(dest)}],
        }
        (target_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest


class AssetManager:
    def __init__(self) -> None:
        self.offline = os.getenv("MONEYOS_OFFLINE", "0") == "1"
        self.sources = [PolyHavenSource(), OpenGameArtSource()]
        self.root = get_short_workdir() / "assets"

    def get_or_download(self, kind: str, query: str) -> tuple[Path, dict]:
        safe_id = shorten_component(query or kind)
        target_dir = safe_join("assets", kind, safe_id)
        manifest_path = target_dir / "manifest.json"
        if manifest_path.exists():
            manifest = verify_manifest_has_license(manifest_path)
            return target_dir, manifest
        if self.offline:
            raise RuntimeError("MONEYOS_OFFLINE=1 and asset not cached")
        for source in self.sources:
            assets = source.list(kind)
            if not assets:
                continue
            manifest = source.download(assets[0], target_dir)
            return target_dir, manifest
        raise RuntimeError(f"No sources for asset kind={kind}")


def asset_id_from_manifest(manifest: dict) -> str:
    payload = json.dumps(manifest, sort_keys=True).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]
