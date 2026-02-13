from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from app.core.net.downloads import DirectUrl, download_from_sources

MIN_VRM_BYTES = 1 * 1024 * 1024
RECEIPT_NAME = ".anime3d_vrm_model.json"

CURATED_VRM_MODELS: tuple[dict[str, Any], ...] = (
    {
        "name": "AliciaSolid",
        "source_urls": [
            "https://raw.githubusercontent.com/vrm-c/UniVRM/master/Tests/Models/Alicia_vrm-0.51/AliciaSolid_vrm-0.51.vrm",
            "https://github.com/vrm-c/vrm-specification/raw/master/samples/VRM1_Constraint_Tests_AliciaSolid.vrm",
        ],
        "license": "sample conditions",
        "filename": "AliciaSolid_vrm-0.51.vrm",
    },
    {
        "name": "AliciaSolidFaceExpression",
        "source_urls": [
            "https://github.com/vrm-c/vrm-specification/raw/master/samples/VRM1_Expression_Tests_AliciaSolid.vrm",
        ],
        "license": "sample conditions",
        "filename": "AliciaSolidFaceExpression.vrm",
    },
    {
        "name": "VRM1FirstPersonA",
        "source_urls": [
            "https://github.com/vrm-c/vrm-specification/raw/master/samples/VRM1_FirstPerson_A.vrm",
        ],
        "license": "sample conditions",
        "filename": "VRM1_FirstPerson_A.vrm",
    },
)


@dataclass(frozen=True)
class ProvisionedCharacter:
    name: str
    source_url: str
    license: str
    downloaded_at: str
    sha256: str
    local_path: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _validate_vrm_file(path: Path) -> None:
    if not path.exists() or path.stat().st_size < MIN_VRM_BYTES:
        raise RuntimeError(f"vrm file too small: {path}")
    head = path.read_bytes()[:512].lower()
    if b"<html" in head or b"<!doctype html" in head:
        raise RuntimeError(f"vrm download returned HTML: {path}")


def _download_file(urls: list[str], target: Path, model_name: str) -> str:
    sources = [DirectUrl(url=url, source="vrm") for url in urls]
    result, source = download_from_sources(
        sources,
        target,
        timeout=120,
        retries=3,
        pack_id="anime3d_vrm_model",
        stage="character_provisioner",
    )
    if not result.ok:
        raise RuntimeError(result.error or f"download failed while installing {model_name}")
    _validate_vrm_file(target)
    return result.final_url or urls[0]


def _licenses_manifest_path(assets_root: Path) -> Path:
    return assets_root / "anime_characters" / "LICENSES" / "third_party_characters.json"


def _vrm_dir(assets_root: Path) -> Path:
    return assets_root / "characters" / "vrm"


def _receipt_path(assets_root: Path) -> Path:
    return _vrm_dir(assets_root) / RECEIPT_NAME


def _validate_manifest(payload: list[dict[str, Any]]) -> None:
    required = {"name", "source_url", "license", "downloaded_at", "sha256", "local_path"}
    for item in payload:
        missing = sorted(required.difference(item.keys()))
        if missing:
            raise RuntimeError(f"Character provenance missing fields {missing} for {item.get('name', 'unknown')}")


def _write_receipt(assets_root: Path, records: list[ProvisionedCharacter], *, ok: bool, error: str | None = None) -> None:
    payload = {
        "ok": ok,
        "pack_id": "anime3d_vrm_model",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "records": [asdict(r) for r in records],
        "error": error,
    }
    receipt = _receipt_path(assets_root)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def provision_anime_characters(assets_root: Path, cache_root: Path) -> list[ProvisionedCharacter]:
    del cache_root
    vrm_dir = _vrm_dir(assets_root)
    licenses_path = _licenses_manifest_path(assets_root)
    vrm_dir.mkdir(parents=True, exist_ok=True)
    licenses_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[ProvisionedCharacter] = []
    errors: list[str] = []
    for spec in CURATED_VRM_MODELS:
        local_file = vrm_dir / spec.get("filename", f"{spec['name']}.vrm")
        source_url = str(spec["source_urls"][0])
        try:
            if not local_file.exists() or local_file.stat().st_size < MIN_VRM_BYTES:
                source_url = _download_file(list(spec["source_urls"]), local_file, str(spec["name"]))
            else:
                _validate_vrm_file(local_file)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"name={spec['name']} error={exc}")
            continue
        sha = _sha256(local_file)
        records.append(
            ProvisionedCharacter(
                name=str(spec["name"]),
                source_url=source_url,
                license=str(spec["license"]),
                downloaded_at=datetime.now(timezone.utc).isoformat(),
                sha256=sha,
                local_path=str(local_file.resolve()),
            )
        )

    if records:
        payload = [asdict(record) for record in records]
        _validate_manifest(payload)
        licenses_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _write_receipt(assets_root, records, ok=True)
    else:
        _write_receipt(assets_root, [], ok=False, error="; ".join(errors) if errors else "no_vrm_models")
    return records


def load_provisioned_characters(assets_root: Path) -> list[ProvisionedCharacter]:
    licenses_path = _licenses_manifest_path(assets_root)
    if not licenses_path.exists():
        return []
    payload = json.loads(licenses_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    _validate_manifest(payload)
    output: list[ProvisionedCharacter] = []
    for item in payload:
        local_path = Path(item["local_path"])
        if not local_path.exists():
            continue
        output.append(ProvisionedCharacter(**item))
    return output
