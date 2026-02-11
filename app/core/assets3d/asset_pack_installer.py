from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import urllib.request
import zipfile
import hashlib

from app.core.visuals.anime_3d.storage import compute_required_bytes, ensure_storage_budget


DEFAULT_ASSET_PACK_URLS: list[str] = [
    "https://github.com/MoneyOS/MoneyOS/releases/latest/download/moneyos-anime3d-starter-pack.zip"
]

_LAST_ASSET_PACK_ERROR: str | None = None


def get_required_anime3d_assets() -> list[str]:
    return [
        "characters/hero.blend",
        "characters/enemy.blend",
        "envs/city.blend",
        "anims/idle.fbx",
        "anims/run.fbx",
        "anims/punch.fbx",
        "vfx/explosion.png",
        "vfx/energy_arc.png",
        "vfx/smoke.png",
    ]


def missing_required_assets(assets_root: Path) -> list[str]:
    missing: list[str] = []
    for rel_path in get_required_anime3d_assets():
        if not (assets_root / rel_path).exists():
            missing.append(rel_path)
    return missing


def _log(message: str, quiet: bool) -> None:
    if quiet:
        print(f"[ASSET_PACK] {message}", flush=True)
        return
    print(f"[ASSET_PACK] {message}", flush=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_with_retries(url: str, dest: Path, retries: int, timeout: int) -> None:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
                dest.write_bytes(response.read())
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(1)
    if last_error:
        raise last_error


def _acquire_lock(lock_path: Path, timeout: int) -> None:
    deadline = time.time() + timeout
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w") as handle:
                handle.write(str(os.getpid()))
            return
        except FileExistsError:
            if time.time() > deadline:
                raise RuntimeError("asset pack lock timeout")
            time.sleep(1)


def _extract_zip(zip_path: Path, extract_dir: Path) -> Path:
    with zipfile.ZipFile(zip_path) as handle:
        handle.extractall(extract_dir)
    entries = [p for p in extract_dir.iterdir() if p.is_dir()]
    root = extract_dir
    if len(entries) == 1:
        candidate = entries[0]
        required = {"characters", "envs", "anims", "vfx"}
        if required.issubset({p.name for p in candidate.iterdir() if p.is_dir()}):
            root = candidate
    return root


def _merge_tree(source: Path, target: Path) -> list[str]:
    installed: list[str] = []
    for item in source.rglob("*"):
        if item.is_dir():
            continue
        rel_path = item.relative_to(source)
        dest = target / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, dest)
        installed.append(str(rel_path))
    return installed


def ensure_anime3d_asset_pack(assets_root: Path, stage: str, strict_assets: bool) -> None:
    global _LAST_ASSET_PACK_ERROR
    missing = missing_required_assets(assets_root)
    if not missing:
        return
    urls_env = os.getenv("MONEYOS_ASSET_PACK_URLS", "")
    urls = [url.strip() for url in urls_env.split(",") if url.strip()] or DEFAULT_ASSET_PACK_URLS
    quiet = os.getenv("MONEYOS_STORAGE_QUIET") == "1"
    if not urls:
        _LAST_ASSET_PACK_ERROR = "no_urls_configured"
        raise RuntimeError(
            "Missing assets and no asset pack URLs configured. "
            "Set MONEYOS_ASSET_PACK_URLS to a starter pack zip."
        )
    timeout = int(os.getenv("MONEYOS_ASSET_PACK_TIMEOUT", "60"))
    retries = int(os.getenv("MONEYOS_ASSET_PACK_RETRIES", "2"))
    estimate = int(os.getenv("MONEYOS_ASSET_PACK_ESTIMATE_BYTES", str(2 * 1024**3)))
    required_bytes = compute_required_bytes(
        estimated_download_size=estimate,
        extraction_overhead=estimate,
    )
    ensure_storage_budget([assets_root], required_bytes, "asset_pack")

    lock_path = assets_root / ".asset_pack.lock"
    _acquire_lock(lock_path, timeout)
    try:
        missing = missing_required_assets(assets_root)
        if not missing:
            return
        temp_root = Path(os.getenv("TEMP") or os.getenv("TMP") or tempfile.gettempdir())
        sha_env = os.getenv("MONEYOS_ASSET_PACK_SHA256")
        success = False
        for url in urls:
            work_dir = Path(tempfile.mkdtemp(prefix="moneyos_asset_pack_", dir=str(temp_root)))
            zip_path = work_dir / "asset_pack.zip"
            try:
                _log(f"downloading {url}", quiet)
                _download_with_retries(url, zip_path, retries, timeout)
                if sha_env:
                    actual = _sha256(zip_path)
                    if actual.lower() != sha_env.lower():
                        message = "asset pack sha256 mismatch"
                        _LAST_ASSET_PACK_ERROR = message
                        if strict_assets:
                            raise RuntimeError(message)
                        _log(f"{message}; trying next URL", quiet)
                        continue
                extract_dir = work_dir / "extract"
                extract_dir.mkdir(parents=True, exist_ok=True)
                root = _extract_zip(zip_path, extract_dir)
                installed = _merge_tree(root, assets_root)
                remaining = missing_required_assets(assets_root)
                if remaining:
                    _LAST_ASSET_PACK_ERROR = "missing_required_files_after_extract"
                    _log(
                        f"installed from {url} but missing {len(remaining)} files; trying next URL",
                        quiet,
                    )
                    continue
                marker = assets_root / ".asset_pack_installed.json"
                marker.write_text(
                    json.dumps(
                        {
                            "url": url,
                            "sha256": sha_env,
                            "timestamp": time.time(),
                            "installed_files": installed,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                _log(f"installed {len(installed)} files from {url}", quiet)
                success = True
                break
            finally:
                shutil.rmtree(work_dir, ignore_errors=True)
        if not success:
            raise RuntimeError("asset pack download failed")
    except Exception as exc:  # noqa: BLE001
        message = (
            "Missing assets even after auto-install attempt. "
            "Check server logs and MONEYOS_ASSET_PACK_URLS."
        )
        _LAST_ASSET_PACK_ERROR = str(exc)
        if quiet:
            print(f"[ASSET_PACK] failed: {message}", flush=True)
        raise RuntimeError(message) from exc
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
    remaining = missing_required_assets(assets_root)
    if remaining:
        message = (
            "Missing assets even after auto-install attempt. "
            "Check server logs and MONEYOS_ASSET_PACK_URLS."
        )
        if strict_assets:
            raise RuntimeError(message)
        raise RuntimeError(message)
