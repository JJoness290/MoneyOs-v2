from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
import urllib.request

from app.core.assets3d.asset_pack_installer import get_required_anime3d_assets
from app.core.visuals.anime_3d.blender_installer import ensure_blender_path
from app.core.paths import get_assets_root

from src.moneyos.auto_assets.downloader import download_url, update_sources_manifest

ANIMATED_HUMAN_URL = "https://opengameart.org/content/animated-human-low-poly"
STREET_PACK_MIRROR_URL = "https://opengameart.org/content/lowpoly-modular-street-pack"
CC0_LICENSE_URL = "https://creativecommons.org/publicdomain/zero/1.0/"
CC0_REQUIRED_ASSETS = [
    "characters/hero.blend",
    "characters/enemy.blend",
    "envs/city.blend",
    "anims/idle.fbx",
    "anims/run.fbx",
    "anims/punch.fbx",
]


class CC0BootstrapError(RuntimeError):
    pass


def _log(message: str) -> None:
    print(f"[AUTO_ASSETS] {message}", flush=True)


def _fetch_html(url: str, timeout: int = 30) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "MoneyOS-CC0-Downloader/1.0"})  # noqa: S310
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return response.read().decode("utf-8", errors="ignore")


def verify_cc0_in_html(html: str) -> bool:
    lowered = html.lower()
    return "cc0" in lowered and "creativecommons.org/publicdomain/zero" in lowered


def extract_oga_zip_url(html: str) -> str | None:
    matches = re.findall(r'href=[\"\']([^\"\']+\.zip)[\"\']', html)
    for match in matches:
        if "sites/default/files" in match:
            return match
    return matches[0] if matches else None


def _resolve_blender_path(blender_path: Path | None) -> Path:
    env_path = os.getenv("BLENDER_PATH") or os.getenv("MONEYOS_BLENDER_PATH")
    if env_path:
        candidate = Path(env_path)
        if candidate.exists():
            return candidate
    if blender_path and blender_path.exists():
        return blender_path
    return ensure_blender_path()


def _run_blender(blender_path: Path, script_path: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    command = [str(blender_path), "--background", "--factory-startup", "--python", str(script_path), "--"]
    command.extend(args)
    return subprocess.run(command, text=True, capture_output=True, check=False)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _temp_dir(cache_root: Path, prefix: str) -> Path:
    temp_root = cache_root / "tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=str(temp_root)))


def _find_files(root: Path, extensions: tuple[str, ...]) -> list[Path]:
    results: list[Path] = []
    for ext in extensions:
        results.extend(root.rglob(f"*{ext}"))
    return sorted(results)


def _choose_animation(files: list[Path], keywords: list[str]) -> Path | None:
    for keyword in keywords:
        for file in files:
            if keyword in file.stem.lower():
                return file
    return None


def _export_actions_to_fbx(
    blender_path: Path,
    source_blend: Path,
    output_dir: Path,
    cache_root: Path,
) -> dict[str, Path]:
    script_path = _temp_dir(cache_root, "moneyos_actions_") / "export_actions.py"
    script_path.write_text(
        """
import argparse
import sys
from pathlib import Path
import bpy

parser = argparse.ArgumentParser()
parser.add_argument("--source", required=True)
parser.add_argument("--output", required=True)
args = sys.argv
if "--" in args:
    args = args[args.index("--") + 1 :]
else:
    args = []
opts = parser.parse_args(args)

bpy.ops.wm.open_mainfile(filepath=opts.source)

output_dir = Path(opts.output)
output_dir.mkdir(parents=True, exist_ok=True)

keywords = {
    "idle": ["idle", "rest"],
    "run": ["run", "walk", "sprint"],
    "punch": ["punch", "attack", "hit"],
}

actions = {action.name.lower(): action for action in bpy.data.actions}
resolved = {}
for label, keys in keywords.items():
    found = None
    for key in keys:
        for name, action in actions.items():
            if key in name:
                found = action
                break
        if found:
            break
    if found:
        resolved[label] = found

if len(resolved) != 3:
    available = ", ".join(sorted(actions.keys()))
    raise SystemExit(f"Missing actions for export. Available actions: {available}")

for label, action in resolved.items():
    bpy.context.scene.frame_start = int(action.frame_range[0])
    bpy.context.scene.frame_end = int(action.frame_range[1])
    bpy.ops.object.select_all(action="SELECT")
    for obj in bpy.context.selected_objects:
        if obj.animation_data:
            obj.animation_data.action = action
    filepath = output_dir / f"{label}.fbx"
    bpy.ops.export_scene.fbx(
        filepath=str(filepath),
        use_selection=False,
        bake_anim=True,
        bake_anim_use_all_actions=False,
        bake_anim_force_startend_keying=True,
        add_leaf_bones=False,
        apply_scale_options='FBX_SCALE_ALL',
        object_types={'ARMATURE', 'MESH'},
    )
""",
        encoding="utf-8",
    )
    result = _run_blender(blender_path, script_path, ["--source", str(source_blend), "--output", str(output_dir)])
    if result.returncode != 0:
        raise CC0BootstrapError(
            "Blender export failed: "
            f"stdout={result.stdout[-400:]} stderr={result.stderr[-400:]}"
        )
    return {
        "idle": output_dir / "idle.fbx",
        "run": output_dir / "run.fbx",
        "punch": output_dir / "punch.fbx",
    }


def _recolor_enemy_blend(
    blender_path: Path,
    source_blend: Path,
    output_blend: Path,
    cache_root: Path,
) -> None:
    script_path = _temp_dir(cache_root, "moneyos_enemy_") / "recolor_enemy.py"
    script_path.write_text(
        """
import argparse
import sys
import bpy

parser = argparse.ArgumentParser()
parser.add_argument("--source", required=True)
parser.add_argument("--output", required=True)
args = sys.argv
if "--" in args:
    args = args[args.index("--") + 1 :]
else:
    args = []
opts = parser.parse_args(args)

bpy.ops.wm.open_mainfile(filepath=opts.source)

for mat in bpy.data.materials:
    if not mat.use_nodes:
        continue
    node = mat.node_tree.nodes.get("Principled BSDF")
    if node:
        node.inputs["Base Color"].default_value = (0.8, 0.2, 0.2, 1)

bpy.ops.wm.save_as_mainfile(filepath=opts.output)
""",
        encoding="utf-8",
    )
    result = _run_blender(blender_path, script_path, ["--source", str(source_blend), "--output", str(output_blend)])
    if result.returncode != 0:
        raise CC0BootstrapError(
            "Blender recolor failed: "
            f"stdout={result.stdout[-400:]} stderr={result.stderr[-400:]}"
        )


def _import_scene_and_save(
    blender_path: Path,
    sources: list[Path],
    output_blend: Path,
    cache_root: Path,
) -> None:
    script_path = _temp_dir(cache_root, "moneyos_scene_") / "assemble_scene.py"
    script_path.write_text(
        f"""
import argparse
import sys
from pathlib import Path
import bpy

parser = argparse.ArgumentParser()
parser.add_argument("--sources", nargs="+", required=True)
parser.add_argument("--output", required=True)
args = sys.argv
if "--" in args:
    args = args[args.index("--") + 1 :]
else:
    args = []
opts = parser.parse_args(args)

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()

for source in opts.sources:
    path = Path(source)
    if path.suffix.lower() == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(path))
    elif path.suffix.lower() == ".obj":
        bpy.ops.import_scene.obj(filepath=str(path))

bpy.ops.object.light_add(type="SUN", location=(0, 0, 10))
bpy.ops.object.camera_add(location=(10, -10, 8), rotation=(1.0, 0.0, 0.7))

bpy.ops.wm.save_as_mainfile(filepath=opts.output)
""",
        encoding="utf-8",
    )
    result = _run_blender(blender_path, script_path, ["--sources", *[str(p) for p in sources], "--output", str(output_blend)])
    if result.returncode != 0:
        raise CC0BootstrapError(
            "Blender scene assembly failed: "
            f"stdout={result.stdout[-400:]} stderr={result.stderr[-400:]}"
        )


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)




def _env_truthy(name: str) -> bool:
    value = os.getenv(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _network_disabled_by_env() -> tuple[bool, str]:
    if _env_truthy("MONEYOS_NO_NETWORK"):
        return True, "MONEYOS_NO_NETWORK=1"
    if _env_truthy("MONEYOS_DISABLE_CC0_BOOTSTRAP"):
        return True, "MONEYOS_DISABLE_CC0_BOOTSTRAP=1"
    providers = (os.getenv("MONEYOS_ASSET_PROVIDERS") or "").strip().lower()
    if providers in {"none", "off", "disabled", "false", "0"}:
        return True, f"MONEYOS_ASSET_PROVIDERS={providers}"
    try:
        max_downloads = int((os.getenv("MONEYOS_ASSET_MAX_DOWNLOADS_PER_RUN") or "").strip() or "-1")
    except ValueError:
        max_downloads = -1
    if max_downloads == 0:
        return True, "MONEYOS_ASSET_MAX_DOWNLOADS_PER_RUN=0"
    return False, ""


def _write_report(report_path: Path, report: dict) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
def ensure_cc0_anime3d_assets(
    assets_root: Path,
    cache_root: Path,
    blender_path: Path | None,
    allow_network: bool = True,
) -> dict:
    start = time.time()
    required = [rel for rel in get_required_anime3d_assets() if rel in CC0_REQUIRED_ASSETS]
    missing = [rel for rel in required if not (assets_root / rel).exists()]
    report_path = cache_root / "cc0_bootstrap_report.json"
    if not missing:
        report = {
            "status": "ok",
            "reason": "all required assets already present",
            "missing": [],
            "installed": [],
            "sources": [],
            "errors": [],
            "duration_seconds": round(time.time() - start, 2),
        }
        _write_report(report_path, report)
        return report

    _log(f"missing={missing}")
    network_disabled, network_reason = _network_disabled_by_env()
    if not allow_network:
        network_disabled = True
        network_reason = network_reason or "allow_network=False"
    if network_disabled:
        report = {
            "status": "skipped",
            "reason": f"CC0 bootstrap skipped: {network_reason}",
            "missing": missing,
            "installed": [],
            "sources": [],
            "errors": [],
            "duration_seconds": round(time.time() - start, 2),
        }
        _write_report(report_path, report)
        return report

    cache_root.mkdir(parents=True, exist_ok=True)
    for subdir in ("characters", "envs", "anims", "vfx"):
        (assets_root / subdir).mkdir(parents=True, exist_ok=True)
    _log(f"assets_root={assets_root}")
    _log(f"cache_root={cache_root}")
    installed: list[str] = []
    sources: list[str] = []
    errors: list[str] = []

    blender_exe = _resolve_blender_path(blender_path)

    try:
        html = _fetch_html(ANIMATED_HUMAN_URL)
        if not verify_cc0_in_html(html):
            raise CC0BootstrapError("Animated Human Low Poly license verification failed (CC0 not found).")
        zip_url = extract_oga_zip_url(html)
        if not zip_url:
            raise CC0BootstrapError("Could not locate Animated Human Low Poly zip download link.")
        if zip_url.startswith("/"):
            zip_url = f"https://opengameart.org{zip_url}"
        _log(f"verifying license for {ANIMATED_HUMAN_URL} => OK")
        _log(f"downloading {zip_url}")
        zip_path, sha256 = download_url(zip_url, cache_root / "downloads")
        update_sources_manifest(cache_root, ANIMATED_HUMAN_URL, "CC0", CC0_LICENSE_URL, sha256, zip_path)
        sources.append(zip_url)
        extract_dir = _temp_dir(cache_root, "moneyos_human_")
        try:
            shutil.unpack_archive(str(zip_path), str(extract_dir))
            blends = _find_files(extract_dir, (".blend",))
            if not blends:
                raise CC0BootstrapError("No .blend files found in Animated Human Low Poly pack.")
            source_blend = blends[0]
            hero_dest = assets_root / "characters" / "hero.blend"
            enemy_dest = assets_root / "characters" / "enemy.blend"
            _ensure_dir(hero_dest)
            shutil.copy2(source_blend, hero_dest)
            installed.append("characters/hero.blend")
            _log(f"installed => {hero_dest}")
            _ensure_dir(enemy_dest)
            _recolor_enemy_blend(blender_exe, hero_dest, enemy_dest, cache_root)
            installed.append("characters/enemy.blend")
            _log(f"installed => {enemy_dest}")

            fbx_files = _find_files(extract_dir, (".fbx",))
            idle = _choose_animation(fbx_files, ["idle"])
            run = _choose_animation(fbx_files, ["run", "walk"])
            punch = _choose_animation(fbx_files, ["punch", "attack"])
            anim_dir = assets_root / "anims"
            anim_dir.mkdir(parents=True, exist_ok=True)
            if idle and run and punch:
                shutil.copy2(idle, anim_dir / "idle.fbx")
                shutil.copy2(run, anim_dir / "run.fbx")
                shutil.copy2(punch, anim_dir / "punch.fbx")
                installed.extend(["anims/idle.fbx", "anims/run.fbx", "anims/punch.fbx"])
                _log("installed => animations from zip")
            else:
                exported = _export_actions_to_fbx(blender_exe, source_blend, anim_dir, cache_root)
                for key, path in exported.items():
                    if not path.exists():
                        raise CC0BootstrapError(f"Missing exported animation {key}.")
                installed.extend(["anims/idle.fbx", "anims/run.fbx", "anims/punch.fbx"])
                _log("installed => animations from Blender export")
        finally:
            shutil.rmtree(extract_dir, ignore_errors=True)

        html = _fetch_html(STREET_PACK_MIRROR_URL)
        if not verify_cc0_in_html(html):
            raise CC0BootstrapError("Street pack license verification failed (CC0 not found).")
        zip_url = extract_oga_zip_url(html)
        if not zip_url:
            raise CC0BootstrapError("Could not locate street pack zip download link.")
        if zip_url.startswith("/"):
            zip_url = f"https://opengameart.org{zip_url}"
        _log(f"verifying license for {STREET_PACK_MIRROR_URL} => OK")
        _log(f"downloading {zip_url}")
        zip_path, sha256 = download_url(zip_url, cache_root / "downloads")
        update_sources_manifest(cache_root, STREET_PACK_MIRROR_URL, "CC0", CC0_LICENSE_URL, sha256, zip_path)
        sources.append(zip_url)
        extract_dir = _temp_dir(cache_root, "moneyos_city_")
        try:
            shutil.unpack_archive(str(zip_path), str(extract_dir))
            blends = _find_files(extract_dir, (".blend",))
            env_dest = assets_root / "envs" / "city.blend"
            if blends:
                _ensure_dir(env_dest)
                shutil.copy2(blends[0], env_dest)
                installed.append("envs/city.blend")
                _log(f"installed => {env_dest}")
            else:
                models = _find_files(extract_dir, (".fbx", ".obj"))
                if not models:
                    raise CC0BootstrapError("Street pack has no .blend/.fbx/.obj assets to build city scene.")
                _ensure_dir(env_dest)
                _import_scene_and_save(blender_exe, models[:25], env_dest, cache_root)
                installed.append("envs/city.blend")
                _log(f"installed => {env_dest}")
        finally:
            shutil.rmtree(extract_dir, ignore_errors=True)

        remaining = [rel for rel in required if not (assets_root / rel).exists()]
        if remaining:
            raise CC0BootstrapError(
                "CC0 bootstrap incomplete. Missing: "
                + ", ".join(remaining)
                + f". Errors: {errors}"
            )

        file_hashes = {}
        for rel in installed:
            file_path = assets_root / rel
            if file_path.exists():
                file_hashes[rel] = _sha256(file_path)
        report = {
            "status": "ok",
            "reason": "downloaded and installed",
            "installed": sorted(set(installed)),
            "file_hashes": file_hashes,
            "sources": sources,
            "missing": [],
            "errors": errors,
            "duration_seconds": round(time.time() - start, 2),
        }
        manifest_path = assets_root / "cc0_manifest.json"
        manifest_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        _write_report(report_path, report)
        _log(f"ready in {report['duration_seconds']}s")
        return report
    except Exception as exc:  # noqa: BLE001
        errors.append(str(exc))
        remaining = [rel for rel in required if not (assets_root / rel).exists()]
        report = {
            "status": "failed",
            "reason": str(exc),
            "installed": sorted(set(installed)),
            "sources": sources,
            "missing": remaining,
            "errors": errors,
            "duration_seconds": round(time.time() - start, 2),
        }
        _write_report(report_path, report)
        raise



if __name__ == "__main__":
    ensure_cc0_anime3d_assets(get_assets_root(), Path("./.cache"), None)
