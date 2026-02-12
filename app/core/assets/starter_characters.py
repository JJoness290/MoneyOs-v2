from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
from urllib.request import urlopen
import zipfile

from app.core.paths import get_characters_dir, get_repo_root

STARTER_PACK_DEFAULT_URL = (
    "https://kenney.nl/media/pages/assets/animated-characters-3/df080ca4ab-1694862585/"
    "kenney_animated-characters-3.zip"
)

USABLE_EXTENSIONS = {".blend", ".fbx", ".glb", ".gltf", ".obj"}


@dataclass(frozen=True)
class StarterInstallResult:
    ok: bool
    installed: bool
    provider: str
    counts: dict[str, int]
    receipt_path: str
    message: str


def _debug_enabled() -> bool:
    return os.getenv("MONEYOS_DEBUG_PHASE3", "0") == "1"


def _log(message: str) -> None:
    if _debug_enabled():
        print(f"[STARTER_CHARPACK] {message}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def list_character_assets(char_dir: Path) -> dict[str, Any]:
    counts: dict[str, int] = {}
    candidates: list[str] = []
    if not char_dir.exists():
        return {"counts": counts, "candidates": candidates, "usable": 0}
    for path in char_dir.rglob("*"):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        counts[ext] = counts.get(ext, 0) + 1
        if ext in USABLE_EXTENSIONS:
            candidates.append(str(path))
    return {"counts": counts, "candidates": candidates, "usable": len(candidates)}


def _min_required_files() -> int:
    try:
        return max(1, int(os.getenv("MONEYOS_STARTER_CHAR_MIN_FILES", "1")))
    except ValueError:
        return 1


def needs_install(char_dir: Path) -> bool:
    inventory = list_character_assets(char_dir)
    return int(inventory.get("usable", 0)) < _min_required_files()


def download_zip(url: str, dest_zip: Path, timeout: tuple[int, int] = (10, 120)) -> None:
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    _log(f"download_start url={url} dest={dest_zip}")
    with urlopen(url, timeout=timeout[1]) as response:  # nosec B310
        total = int(response.headers.get("Content-Length") or 0)
        read = 0
        with dest_zip.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 128)
                if not chunk:
                    break
                handle.write(chunk)
                read += len(chunk)
                if _debug_enabled() and total > 0:
                    pct = (read / total) * 100
                    if int(pct) % 20 == 0:
                        _log(f"download_progress pct={pct:.1f}")
    _log("download_done")


def _flatten_root(extracted_root: Path) -> Path:
    children = [child for child in extracted_root.iterdir()]
    if len(children) == 1 and children[0].is_dir():
        return children[0]
    return extracted_root


def _copy_tree(src_root: Path, dst_root: Path) -> int:
    installed_files = 0
    dst_root.mkdir(parents=True, exist_ok=True)
    for source in src_root.rglob("*"):
        if not source.is_file():
            continue
        rel = source.relative_to(src_root)
        target = dst_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            continue
        shutil.copy2(source, target)
        installed_files += 1
    return installed_files


def _receipt_path(char_dir: Path) -> Path:
    return char_dir / ".starter_pack.json"


def _legacy_candidate_dirs() -> list[Path]:
    repo_root = get_repo_root()
    candidates = [
        (repo_root / "assets" / "characters_3d").resolve(),
        (repo_root / "assets" / "characters").resolve(),
    ]
    env_char_dir = os.getenv("MONEYOS_CHARACTERS_DIR")
    if env_char_dir:
        candidates.append((repo_root / env_char_dir).resolve())
    unique: list[Path] = []
    for item in candidates:
        if item not in unique:
            unique.append(item)
    return unique


def _migrate_legacy_if_needed(runtime_char_dir: Path) -> int:
    runtime_inventory = list_character_assets(runtime_char_dir)
    if int(runtime_inventory.get("usable", 0)) >= _min_required_files():
        return 0
    copied_total = 0
    for legacy_dir in _legacy_candidate_dirs():
        if legacy_dir == runtime_char_dir or not legacy_dir.exists():
            continue
        legacy_inventory = list_character_assets(legacy_dir)
        if int(legacy_inventory.get("usable", 0)) <= 0:
            continue
        target = runtime_char_dir / "starter_pack"
        copied = _copy_tree(legacy_dir, target)
        if copied > 0:
            copied_total += copied
            print(
                f"PHASE3_CHARPACK_MIGRATED from={legacy_dir} to={runtime_char_dir} copied={copied}"
            )
    return copied_total


def _write_receipt(
    char_dir: Path,
    *,
    provider: str,
    url: str,
    archive_sha256: str,
    files_installed: int,
) -> Path:
    receipt_path = _receipt_path(char_dir)
    inventory = list_character_assets(char_dir)
    payload = {
        "provider": provider,
        "url": url,
        "installed_at": datetime.now(timezone.utc).isoformat(),
        "archive_sha256": archive_sha256,
        "files_installed": files_installed,
        "usable_assets": inventory.get("usable", 0),
        "counts": inventory.get("counts", {}),
        "starter_pack_path": str((char_dir / "starter_pack").resolve()),
        "attribution": {
            "name": "Kenney Animated Characters 3",
            "license": "CC0",
            "source": "https://kenney.nl/assets/animated-characters-3",
        },
        "version": 1,
    }
    receipt_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return receipt_path


def ensure_starter_characters_installed(char_dir: Path | None = None, strict: bool = False) -> dict[str, Any]:
    runtime_char_dir = (char_dir or get_characters_dir()).resolve()
    if _debug_enabled():
        print(f"PHASE3_CHARPACK_TARGET char_dir={runtime_char_dir}")
    provider = os.getenv("MONEYOS_STARTER_CHAR_PACK_PROVIDER", "kenney_animated_characters_3").strip()
    pack_url = os.getenv("MONEYOS_STARTER_CHAR_PACK_URL", STARTER_PACK_DEFAULT_URL).strip()
    expected_sha = os.getenv("MONEYOS_STARTER_CHAR_PACK_SHA256", "").strip().lower()
    auto_install = os.getenv("MONEYOS_AUTO_INSTALL_STARTER_CHARACTERS", "1") == "1"

    runtime_char_dir.mkdir(parents=True, exist_ok=True)
    migrated_files = _migrate_legacy_if_needed(runtime_char_dir)
    pre = list_character_assets(runtime_char_dir)
    if int(pre.get("usable", 0)) >= _min_required_files():
        receipt = _receipt_path(runtime_char_dir)
        if _debug_enabled():
            print(f"PHASE3_CHARPACK_RECEIPT receipt_path={receipt}")
        return {
            "ok": True,
            "installed": False,
            "provider": provider,
            "counts": pre.get("counts", {}),
            "receipt_path": str(receipt.resolve()),
            "message": "starter characters already available",
            "migrated_files": migrated_files,
        }

    if not auto_install:
        message = (
            "No usable character assets found and auto-install is disabled. "
            "Set MONEYOS_AUTO_INSTALL_STARTER_CHARACTERS=1 to install starter characters automatically."
        )
        if strict:
            raise RuntimeError(message)
        return {
            "ok": False,
            "installed": False,
            "provider": provider,
            "counts": pre.get("counts", {}),
            "receipt_path": str(_receipt_path(runtime_char_dir).resolve()),
            "message": message,
        }

    with tempfile.TemporaryDirectory(prefix="moneyos_starter_char_") as tmpdir:
        tmp_root = Path(tmpdir)
        zip_path = tmp_root / "starter_pack.zip"
        extract_path = tmp_root / "extract"
        extract_path.mkdir(parents=True, exist_ok=True)

        download_zip(pack_url, zip_path)
        archive_sha = sha256_file(zip_path)
        if expected_sha and archive_sha.lower() != expected_sha:
            raise RuntimeError(
                "Starter character pack SHA256 mismatch. "
                f"expected={expected_sha} actual={archive_sha}"
            )

        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(extract_path)
        flattened = _flatten_root(extract_path)

        target_pack_root = runtime_char_dir / "starter_pack"
        installed_files = _copy_tree(flattened, target_pack_root)

        # promote license/readme if available
        for candidate in target_pack_root.rglob("*"):
            if not candidate.is_file():
                continue
            name = candidate.name.lower()
            if name.startswith("license"):
                license_target = target_pack_root / "LICENSE.txt"
                if not license_target.exists():
                    shutil.copy2(candidate, license_target)
            if name.startswith("readme"):
                readme_target = target_pack_root / "README.txt"
                if not readme_target.exists():
                    shutil.copy2(candidate, readme_target)

    receipt = _write_receipt(
        runtime_char_dir,
        provider=provider,
        url=pack_url,
        archive_sha256=archive_sha,
        files_installed=installed_files,
    )
    if _debug_enabled():
        print(f"PHASE3_CHARPACK_RECEIPT receipt_path={receipt}")
    post = list_character_assets(runtime_char_dir)
    if int(post.get("usable", 0)) < _min_required_files():
        raise RuntimeError(
            "Starter character pack installed but no usable assets were found. "
            f"usable={post.get('usable', 0)} required={_min_required_files()}"
        )

    return {
        "ok": True,
        "installed": True,
        "provider": provider,
        "counts": post.get("counts", {}),
        "receipt_path": str(receipt.resolve()),
        "message": "starter characters installed",
        "migrated_files": migrated_files,
    }
