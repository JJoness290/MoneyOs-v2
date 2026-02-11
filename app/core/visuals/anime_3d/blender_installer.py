from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import urllib.request
import zipfile

from app.core.paths import get_assets_root

BLENDER_CONFIG_DIR = get_assets_root() / "tools" / "blender"
BLENDER_PATH_FILE = BLENDER_CONFIG_DIR / "blender_path.txt"
BLENDER_DOWNLOAD_URL = os.getenv(
    "MONEYOS_BLENDER_URL",
    "https://download.blender.org/release/Blender4.1/blender-4.1.1-windows-x64.zip",
)
BLENDER_MIN_BYTES = int(os.getenv("MONEYOS_BLENDER_MIN_BYTES", str(200 * 1024 * 1024)))


def _candidate_paths() -> list[Path]:
    base_dir = Path("C:/Program Files/Blender Foundation")
    local_app = Path(os.getenv("LOCALAPPDATA", "")) / "Programs" / "Blender Foundation"
    candidates = [
        base_dir / "Blender" / "blender.exe",
        base_dir / "Blender 4.0" / "blender.exe",
        base_dir / "Blender 3.6" / "blender.exe",
        local_app / "Blender" / "blender.exe",
    ]
    if base_dir.exists():
        for path in sorted(base_dir.glob("Blender*/blender.exe")):
            candidates.append(path)
    if local_app.exists():
        for path in sorted(local_app.glob("Blender*/blender.exe")):
            candidates.append(path)
    return candidates


def _read_config_path() -> Path | None:
    if not BLENDER_PATH_FILE.exists():
        return None
    path = Path(BLENDER_PATH_FILE.read_text(encoding="utf-8").strip())
    return path if path.exists() else None


def _write_config_path(path: Path) -> None:
    BLENDER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    BLENDER_PATH_FILE.write_text(str(path), encoding="utf-8")


def _download_blender(zip_path: Path) -> None:
    BLENDER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(BLENDER_DOWNLOAD_URL) as response:  # noqa: S310
        data = response.read()
    if len(data) < BLENDER_MIN_BYTES:
        raise RuntimeError("Blender download too small; aborting")
    zip_path.write_bytes(data)


def _extract_blender(zip_path: Path, target_dir: Path) -> Path:
    with zipfile.ZipFile(zip_path) as handle:
        handle.extractall(target_dir)
    for candidate in target_dir.rglob("blender.exe"):
        return candidate
    raise RuntimeError("Blender extraction completed but blender.exe not found")


def ensure_blender_path() -> Path:
    env_path = os.getenv("MONEYOS_BLENDER_PATH")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path
    config_path = _read_config_path()
    if config_path:
        return config_path
    for candidate in _candidate_paths():
        if candidate.exists():
            _write_config_path(candidate)
            os.environ.setdefault("MONEYOS_BLENDER_PATH", str(candidate))
            return candidate
    zip_path = BLENDER_CONFIG_DIR / "blender.zip"
    install_root = BLENDER_CONFIG_DIR / "install"
    if install_root.exists():
        shutil.rmtree(install_root, ignore_errors=True)
    try:
        _download_blender(zip_path)
        blender_exe = _extract_blender(zip_path, install_root)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Blender required; auto-install failed. Set MONEYOS_BLENDER_PATH to blender.exe."
        ) from exc
    _write_config_path(blender_exe)
    os.environ.setdefault("MONEYOS_BLENDER_PATH", str(blender_exe))
    return blender_exe


def blender_status() -> dict[str, str]:
    payload = {
        "source": "env" if os.getenv("MONEYOS_BLENDER_PATH") else "auto",
        "path": os.getenv("MONEYOS_BLENDER_PATH", ""),
        "download_url": BLENDER_DOWNLOAD_URL,
    }
    return json.loads(json.dumps(payload))
