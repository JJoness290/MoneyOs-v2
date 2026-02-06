from __future__ import annotations

import hashlib
import json
from pathlib import Path


def verify_manifest_has_license(manifest_path: Path) -> dict:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not payload.get("license"):
        raise RuntimeError("Asset manifest missing license")
    return payload


def verify_sha256(file_path: Path, expected: str) -> None:
    digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
    if digest != expected:
        raise RuntimeError(f"SHA256 mismatch for {file_path.name}")


def verify_files_exist(manifest: dict) -> None:
    files = manifest.get("files", [])
    for entry in files:
        path = Path(entry.get("path", ""))
        if not path.exists():
            raise RuntimeError(f"Missing asset file: {path}")
