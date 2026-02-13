from __future__ import annotations

import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile
import time
import zipfile

from PIL import Image, ImageDraw, ImageFilter

from app.core.assets3d.asset_pack_installer import (
    get_required_anime3d_assets as _get_required_anime3d_assets,
    missing_required_assets as _missing_required_assets,
)
from app.core.net.downloads import DirectUrl, download_from_sources
from app.core.paths import get_output_root
from app.core.visuals.anime_3d.blender_installer import ensure_blender_path
from app.core.visuals.anime_3d.blender_runner import BlenderCommand, run_blender
from src.moneyos.auto_assets.cc0_bootstrap_anime3d import ensure_cc0_anime3d_assets


KENNEY_VFX_ZIP_URLS = [
    "https://kenney.nl/media/pages/assets/particle-pack/1.0/particle-pack.zip",
    "https://kenney.nl/media/pages/assets/particle-pack/1.1/particle-pack.zip",
]

_LAST_AUTO_ASSETS_ERROR: str | None = None


def get_required_anime3d_assets() -> list[str]:
    return _get_required_anime3d_assets()


def missing_required_assets(assets_root: Path) -> list[str]:
    return _missing_required_assets(assets_root)


def get_last_auto_assets_error() -> str | None:
    return _LAST_AUTO_ASSETS_ERROR


def _log(message: str, quiet: bool) -> None:
    if quiet:
        return
    print(f"[AUTO_ASSETS] {message}", flush=True)


def _download_with_retries(url: str, dest: Path, retries: int, timeout: int, stage: str) -> None:
    result, _ = download_from_sources(
        [DirectUrl(url=url, source="vfx")],
        dest,
        timeout=timeout,
        retries=max(1, retries + 1),
        pack_id="anime3d_vfx",
        stage=stage,
    )
    if not result.ok:
        raise RuntimeError(result.error or f"download failed for {url}")


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
                raise RuntimeError("auto assets lock timeout")
            time.sleep(1)


def _choose_png_by_keyword(paths: list[Path], keywords: list[str]) -> Path | None:
    lowered = {path: path.name.lower() for path in paths}
    for keyword in keywords:
        for path, name in lowered.items():
            if keyword in name:
                return path
    return None


def _collect_alpha_pngs(root: Path) -> list[Path]:
    candidates: list[Path] = []
    for path in root.rglob("*.png"):
        try:
            with Image.open(path) as image:
                if image.mode in {"RGBA", "LA"} or (
                    image.mode == "P" and "transparency" in image.info
                ):
                    candidates.append(path)
        except Exception:  # noqa: BLE001
            continue
    return candidates


def _download_vfx_sprites(
    assets_root: Path,
    timeout: int,
    retries: int,
    quiet: bool,
    errors: list[str],
    sources: list[str],
) -> list[str]:
    vfx_dir = assets_root / "vfx"
    vfx_dir.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix="moneyos_vfx_"))
    installed: list[str] = []
    try:
        zip_path = temp_root / "vfx.zip"
        downloaded = False
        for url in KENNEY_VFX_ZIP_URLS:
            try:
                _log(f"downloading vfx sprites from {url}", quiet)
                _download_with_retries(url, zip_path, retries, timeout, stage="vfx_sprites")
                sources.append(url)
                downloaded = True
                break
            except Exception as exc:  # noqa: BLE001
                errors.append(f"vfx download failed {url}: {exc}")
        if not downloaded:
            raise RuntimeError("no vfx download succeeded")
        extract_dir = temp_root / "extract"
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as handle:
            handle.extractall(extract_dir)
        pngs = _collect_alpha_pngs(extract_dir)
        if not pngs:
            raise RuntimeError("no alpha PNGs found in VFX pack")
        explosion = _choose_png_by_keyword(pngs, ["explosion", "blast", "burst"]) or pngs[0]
        energy = _choose_png_by_keyword(pngs, ["energy", "lightning", "spark"]) or (
            pngs[1] if len(pngs) > 1 else pngs[0]
        )
        smoke = _choose_png_by_keyword(pngs, ["smoke", "fog", "cloud"]) or (
            pngs[2] if len(pngs) > 2 else pngs[-1]
        )
        mapping = {
            "vfx/explosion.png": explosion,
            "vfx/energy_arc.png": energy,
            "vfx/smoke.png": smoke,
        }
        for rel_path, source in mapping.items():
            target = assets_root / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            installed.append(rel_path)
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
    return installed


def _generate_explosion(path: Path, size: int = 256) -> None:
    rng = random.Random(42)
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    center = size // 2
    for _ in range(180):
        angle = rng.uniform(0, 360)
        length = rng.uniform(size * 0.2, size * 0.48)
        width = rng.uniform(2, 6)
        x = center + length * math.cos(math.radians(angle))
        y = center + length * math.sin(math.radians(angle))
        draw.line(
            [(center, center), (x, y)],
            fill=(255, 180, 60, int(rng.uniform(120, 200))),
            width=int(width),
        )
    draw.ellipse(
        (center - 40, center - 40, center + 40, center + 40),
        fill=(255, 220, 140, 200),
    )
    image = image.filter(ImageFilter.GaussianBlur(radius=4))
    image.save(path, "PNG")


def _generate_energy_arc(path: Path, size: int = 256) -> None:
    rng = random.Random(7)
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    points = []
    x = 20
    y = size // 2
    for _ in range(8):
        x += rng.randint(20, 35)
        y += rng.randint(-20, 20)
        points.append((x, y))
    points = [(20, size // 2)] + points + [(size - 20, size // 2)]
    draw.line(points, fill=(120, 200, 255, 220), width=4)
    draw.line(points, fill=(180, 240, 255, 180), width=2)
    image = image.filter(ImageFilter.GaussianBlur(radius=1.5))
    image.save(path, "PNG")


def _generate_smoke(path: Path, size: int = 256) -> None:
    rng = random.Random(13)
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    for _ in range(12):
        radius = rng.randint(20, 50)
        x = rng.randint(radius, size - radius)
        y = rng.randint(radius, size - radius)
        alpha = rng.randint(40, 90)
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(200, 200, 200, alpha),
        )
    image = image.filter(ImageFilter.GaussianBlur(radius=8))
    image.save(path, "PNG")


def _generate_vfx_procedural(assets_root: Path) -> list[str]:
    vfx_dir = assets_root / "vfx"
    vfx_dir.mkdir(parents=True, exist_ok=True)
    explosion_path = vfx_dir / "explosion.png"
    energy_path = vfx_dir / "energy_arc.png"
    smoke_path = vfx_dir / "smoke.png"
    _generate_explosion(explosion_path)
    _generate_energy_arc(energy_path)
    _generate_smoke(smoke_path)
    return ["vfx/explosion.png", "vfx/energy_arc.png", "vfx/smoke.png"]


def _generate_blender_assets(assets_root: Path, quiet: bool) -> None:
    script_path = (Path(__file__).parent / "blender" / "procedural_assets.py").resolve()
    command = BlenderCommand(
        script_path=script_path,
        args=["--assets-dir", str(assets_root)],
    )
    _log("running blender procedural asset generation", quiet)
    run_blender(command)


def ensure_anime3d_assets_auto(assets_root: Path, stage: str, strict_assets: bool) -> None:
    global _LAST_AUTO_ASSETS_ERROR
    missing = missing_required_assets(assets_root)
    if not missing:
        return
    quiet = os.getenv("MONEYOS_STORAGE_QUIET") == "1"
    cache_root = get_output_root() / "auto_assets"
    _log(f"assets_root={assets_root}", quiet)
    _log(f"cache_root={cache_root}", quiet)
    timeout = int(os.getenv("MONEYOS_AUTO_ASSETS_TIMEOUT", "60"))
    retries = int(os.getenv("MONEYOS_AUTO_ASSETS_RETRIES", "2"))
    lock_path = assets_root / ".auto_assets.lock"
    _acquire_lock(lock_path, timeout)
    errors: list[str] = []
    sources: list[str] = []
    installed_files: list[str] = []
    procedural_files: list[str] = []
    missing_before = missing
    try:
        missing = missing_required_assets(assets_root)
        if not missing:
            return
        allow_network = os.getenv("MONEYOS_DISABLE_NET") != "1"
        cc0_disabled = os.getenv("MONEYOS_DISABLE_CC0_BOOTSTRAP") == "1"
        no_network = os.getenv("MONEYOS_NO_NETWORK") == "1"
        if cc0_disabled or no_network:
            _log("[BOOTSTRAP] Using local assets only", quiet)
        else:
            try:
                ensure_cc0_anime3d_assets(
                    assets_root,
                    cache_root,
                    ensure_blender_path(),
                    allow_network=allow_network,
                )
            except Exception as exc:  # noqa: BLE001
                warning = f"cc0 bootstrap failed (continuing with local/procedural assets): {exc}"
                errors.append(warning)
                _log(warning, quiet)
        missing = missing_required_assets(assets_root)
        missing_set = set(missing)
        if any(path.startswith("vfx/") for path in missing_set):
            try:
                installed_files += _download_vfx_sprites(
                    assets_root,
                    timeout,
                    retries,
                    quiet,
                    errors,
                    sources,
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"vfx download failed: {exc}")
                installed_files += _generate_vfx_procedural(assets_root)
                procedural_files += [
                    "vfx/explosion.png",
                    "vfx/energy_arc.png",
                    "vfx/smoke.png",
                ]
        non_vfx_missing = [
            path
            for path in missing_required_assets(assets_root)
            if not path.startswith("vfx/")
        ]
        if non_vfx_missing:
            errors.append("non-vfx assets require procedural Blender generation")
            _generate_blender_assets(assets_root, quiet)
            procedural_files += non_vfx_missing
        remaining = missing_required_assets(assets_root)
        marker = assets_root / ".auto_assets_installed.json"
        marker.write_text(
            json.dumps(
                {
                    "timestamp": time.time(),
                    "stage": stage,
                    "missing_before": missing_before,
                    "installed_files": sorted(set(installed_files)),
                    "sources": sources,
                    "procedural": sorted(set(procedural_files)),
                    "errors": errors,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if remaining:
            message = "Auto asset bootstrap incomplete. Missing: " + ", ".join(remaining)
            _LAST_AUTO_ASSETS_ERROR = message
            if strict_assets:
                raise RuntimeError(message)
            _log(f"warning_non_strict {message}", quiet)
        _log("auto assets ready", quiet)
    except Exception as exc:  # noqa: BLE001
        _LAST_AUTO_ASSETS_ERROR = str(exc)
        if strict_assets:
            raise
        _log(f"non_strict_continue error={exc}", quiet)
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
