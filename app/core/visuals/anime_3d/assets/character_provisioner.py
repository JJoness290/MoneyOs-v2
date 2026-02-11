from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.request import urlopen

CURATED_VRM_MODELS: tuple[dict[str, str], ...] = (
    {
        "name": "AliciaSolid",
        "source_url": "https://github.com/vrm-c/vrm-specification/raw/master/samples/VRM1_Constraint_Tests_AliciaSolid.vrm",
        "license": "sample conditions",
    },
    {
        "name": "AliciaSolidFaceExpression",
        "source_url": "https://github.com/vrm-c/vrm-specification/raw/master/samples/VRM1_Expression_Tests_AliciaSolid.vrm",
        "license": "sample conditions",
    },
    {
        "name": "VRM1FirstPersonA",
        "source_url": "https://github.com/vrm-c/vrm-specification/raw/master/samples/VRM1_FirstPerson_A.vrm",
        "license": "sample conditions",
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


def _download_file(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=120) as response:  # nosec B310
        data = response.read()
    target.write_bytes(data)


def _licenses_manifest_path(assets_root: Path) -> Path:
    return assets_root / "anime_characters" / "LICENSES" / "third_party_characters.json"


def _vrm_dir(assets_root: Path) -> Path:
    return assets_root / "anime_characters" / "vrm"


def _validate_manifest(payload: list[dict[str, Any]]) -> None:
    required = {"name", "source_url", "license", "downloaded_at", "sha256", "local_path"}
    for item in payload:
        missing = sorted(required.difference(item.keys()))
        if missing:
            raise RuntimeError(f"Character provenance missing fields {missing} for {item.get('name', 'unknown')}")


def provision_anime_characters(assets_root: Path, cache_root: Path) -> list[ProvisionedCharacter]:
    del cache_root
    vrm_dir = _vrm_dir(assets_root)
    licenses_path = _licenses_manifest_path(assets_root)
    vrm_dir.mkdir(parents=True, exist_ok=True)
    licenses_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[ProvisionedCharacter] = []
    for spec in CURATED_VRM_MODELS:
        local_file = vrm_dir / f"{spec['name']}.vrm"
        if not local_file.exists() or local_file.stat().st_size < 512:
            _download_file(spec["source_url"], local_file)
        sha = _sha256(local_file)
        records.append(
            ProvisionedCharacter(
                name=spec["name"],
                source_url=spec["source_url"],
                license=spec["license"],
                downloaded_at=datetime.now(timezone.utc).isoformat(),
                sha256=sha,
                local_path=str(local_file.resolve()),
            )
        )

    payload = [asdict(record) for record in records]
    _validate_manifest(payload)
    licenses_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return records


def load_provisioned_characters(assets_root: Path) -> list[ProvisionedCharacter]:
    licenses_path = _licenses_manifest_path(assets_root)
    if not licenses_path.exists():
        return []
    payload = json.loads(licenses_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError("Character provenance manifest is invalid; expected list")
    _validate_manifest(payload)
    output: list[ProvisionedCharacter] = []
    for item in payload:
        local_path = Path(item["local_path"])
        if not local_path.exists():
            raise RuntimeError(f"Provisioned character missing from disk: {local_path}")
        output.append(ProvisionedCharacter(**item))
    return output
