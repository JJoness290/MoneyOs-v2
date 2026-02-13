from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any
from urllib.parse import urljoin
import zipfile

import requests

from app.core.paths import get_characters_dir

PACK_NAME = "kenney_animated_characters_3"
PACK_DIR_NAME = "kenney_animated_characters_3"
RECEIPT_NAME = ".starter_pack.json"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36 MoneyOS/Phase3"
)

SOURCE_PAGES: list[tuple[str, str]] = [
    ("kenney.nl", "https://kenney.nl/assets/animated-characters-3"),
    ("itch.io", "https://kenney-assets.itch.io/animated-characters-3"),
    ("opengameart", "https://opengameart.org/content/animated-human-low-poly"),
]

USABLE_EXTENSIONS = {".blend", ".fbx", ".glb", ".gltf", ".obj"}
RIGGED_EXTENSIONS = {".fbx", ".glb", ".gltf"}


def _debug_enabled() -> bool:
    return os.getenv("MONEYOS_DEBUG_PHASE3", "0") == "1"


def _log(message: str) -> None:
    print(f"[STARTER_CHARPACK] {message}")


def _receipt_path(char_dir: Path) -> Path:
    return char_dir / RECEIPT_NAME


def _pack_dir(char_dir: Path) -> Path:
    return char_dir / PACK_DIR_NAME


def list_character_assets(char_dir: Path) -> dict[str, Any]:
    counts: dict[str, int] = {}
    rigged_counts: dict[str, int] = {}
    candidates: list[str] = []
    rigged_candidates: list[str] = []
    if not char_dir.exists():
        return {
            "counts": {},
            "rigged_counts": {},
            "candidates": [],
            "rigged_candidates": [],
            "usable": 0,
            "rigged_usable": 0,
        }
    for path in char_dir.rglob("*"):
        if not path.is_file() or path.name.lower().endswith(".disabled"):
            continue
        ext = path.suffix.lower()
        counts[ext] = counts.get(ext, 0) + 1
        if ext in USABLE_EXTENSIONS:
            candidates.append(str(path))
        if ext in RIGGED_EXTENSIONS:
            rigged_counts[ext] = rigged_counts.get(ext, 0) + 1
            rigged_candidates.append(str(path))
    return {
        "counts": counts,
        "rigged_counts": rigged_counts,
        "candidates": candidates,
        "rigged_candidates": rigged_candidates,
        "usable": len(candidates),
        "rigged_usable": len(rigged_candidates),
    }


def _min_required_rigged() -> int:
    try:
        return max(1, int(os.getenv("MONEYOS_STARTER_CHAR_MIN_RIGGED", "1")))
    except ValueError:
        return 1


def _download_enabled() -> bool:
    if os.getenv("MONEYOS_AUTO_INSTALL_STARTER_CHARACTERS", "1") != "1":
        return False
    if os.getenv("MONEYOS_NO_NETWORK", "0") == "1" or os.getenv("MONEYOS_DISABLE_NET", "0") == "1":
        return False
    if os.getenv("MONEYOS_DISABLE_CC0_BOOTSTRAP", "0") == "1":
        return False
    providers = (os.getenv("MONEYOS_ASSET_PROVIDERS") or "").strip().lower()
    if providers in {"none", "off", "disabled", "false", "0"}:
        return False
    try:
        max_dl = int((os.getenv("MONEYOS_ASSET_MAX_DOWNLOADS_PER_RUN") or "").strip() or "-1")
    except ValueError:
        max_dl = -1
    return max_dl != 0


def _extract_zip_link(page_url: str, html: str) -> str | None:
    matches = re.findall(r"href=['\"]([^'\"]+\.zip(?:\?[^'\"]*)?)['\"]", html, flags=re.IGNORECASE)
    if not matches:
        return None
    preferred = None
    for raw in matches:
        full = urljoin(page_url, raw)
        lowered = full.lower()
        if "animated" in lowered or "character" in lowered or "kenney" in lowered:
            preferred = full
            break
    return preferred or urljoin(page_url, matches[0])


def _download_zip_multi_source(cache_zip: Path) -> tuple[Path, str]:
    cache_zip.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "text/html,application/zip,*/*"})

    for source_name, source_url in SOURCE_PAGES:
        for attempt in range(1, 4):
            try:
                _log(f"download_source={source_name} attempt={attempt} page={source_url}")
                page_resp = session.get(source_url, timeout=60, allow_redirects=True)
                page_resp.raise_for_status()
                ctype = (page_resp.headers.get("Content-Type") or "").lower()
                if "zip" in ctype and ".zip" in page_resp.url.lower():
                    zip_url = page_resp.url
                else:
                    zip_url = _extract_zip_link(page_resp.url, page_resp.text)
                if not zip_url:
                    raise RuntimeError("zip_link_not_found")

                _log(f"download_zip source={source_name} url={zip_url}")
                with session.get(zip_url, timeout=60, allow_redirects=True, stream=True) as zip_resp:
                    zip_resp.raise_for_status()
                    zctype = (zip_resp.headers.get("Content-Type") or "").lower()
                    if "text/html" in zctype:
                        raise RuntimeError("zip_url_returned_html")
                    with cache_zip.open("wb") as handle:
                        for chunk in zip_resp.iter_content(chunk_size=1024 * 128):
                            if chunk:
                                handle.write(chunk)
                if cache_zip.stat().st_size <= 0:
                    raise RuntimeError("empty_zip_download")
                return cache_zip, source_name
            except Exception as exc:  # noqa: BLE001
                _log(f"download_retry source={source_name} attempt={attempt} error={exc}")
                if attempt == 3:
                    _log(f"download_source_failed source={source_name}")
                continue
    raise RuntimeError("all_sources_failed")


def _flatten_root(extracted_root: Path) -> Path:
    children = [child for child in extracted_root.iterdir()]
    if len(children) == 1 and children[0].is_dir():
        return children[0]
    return extracted_root


def _write_receipt(char_dir: Path, *, ok: bool, installed: bool, source: str, message: str) -> Path:
    receipt = {
        "ok": ok,
        "installed": installed,
        "pack": PACK_NAME,
        "source": source,
        "installed_at": datetime.now(timezone.utc).isoformat(),
        "message": message,
    }
    receipt_path = _receipt_path(char_dir)
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt_path


def ensure_charpack_installed(assets_root: Path) -> Path:
    char_dir = (assets_root / "characters").resolve()
    char_dir.mkdir(parents=True, exist_ok=True)
    pack_dir = _pack_dir(char_dir)
    receipt_path = _receipt_path(char_dir)

    # cache hit
    inv = list_character_assets(char_dir)
    if receipt_path.exists() and pack_dir.exists() and int(inv.get("rigged_usable", 0)) >= _min_required_rigged():
        _log("cache_hit receipt_exists=1")
        return pack_dir

    # corruption detection + auto-repair
    if pack_dir.exists() and int(inv.get("usable", 0)) == 0:
        _log(f"corrupt_pack_detected deleting={pack_dir}")
        shutil.rmtree(pack_dir, ignore_errors=True)

    if not _download_enabled():
        _log("download_skipped offline_or_disabled=1")
        pack_dir.mkdir(parents=True, exist_ok=True)
        _write_receipt(char_dir, ok=True, installed=False, source="local_fallback", message="offline/disabled")
        return pack_dir

    cache_zip = char_dir / ".cache" / f"{PACK_NAME}.zip"
    source_used = "unknown"
    try:
        zip_path, source_used = _download_zip_multi_source(cache_zip)
        with tempfile.TemporaryDirectory(prefix="moneyos_charpack_") as tmp:
            tmp_root = Path(tmp)
            extract_dir = tmp_root / "extract"
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as archive:
                archive.extractall(extract_dir)
            flattened = _flatten_root(extract_dir)
            if pack_dir.exists():
                shutil.rmtree(pack_dir, ignore_errors=True)
            pack_dir.mkdir(parents=True, exist_ok=True)
            for source in flattened.rglob("*"):
                if not source.is_file():
                    continue
                target = pack_dir / source.relative_to(flattened)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        _write_receipt(char_dir, ok=True, installed=True, source=source_used, message="installed")
        _log(f"install_success source={source_used} pack_dir={pack_dir}")
    except Exception as exc:  # noqa: BLE001
        _log(f"install_failed error={exc} fallback=procedural")
        pack_dir.mkdir(parents=True, exist_ok=True)
        _write_receipt(char_dir, ok=True, installed=False, source="procedural_fallback", message=str(exc))

    return pack_dir


def ensure_starter_characters_installed(char_dir: Path | None = None, strict: bool = False) -> dict[str, Any]:
    del strict
    runtime_char_dir = (char_dir or get_characters_dir()).resolve()
    assets_root = runtime_char_dir.parent
    pack_dir = ensure_charpack_installed(assets_root)
    inventory = list_character_assets(runtime_char_dir)
    receipt = _receipt_path(runtime_char_dir)
    payload: dict[str, Any] = {
        "ok": True,
        "installed": False,
        "provider": PACK_NAME,
        "source": "unknown",
        "counts": inventory.get("counts", {}),
        "rigged_count": inventory.get("rigged_usable", 0),
        "required_min_rigged": _min_required_rigged(),
        "receipt_path": str(receipt.resolve()),
        "starter_pack_path": str(pack_dir.resolve()),
        "reason": "ready",
        "message": "starter characters ready",
    }
    if receipt.exists():
        try:
            receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
            payload["installed"] = bool(receipt_payload.get("installed", False))
            payload["source"] = receipt_payload.get("source", "unknown")
            payload["reason"] = receipt_payload.get("message", payload["reason"])
        except json.JSONDecodeError:
            pass
    return payload
