from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from app.core.net.downloads import DirectUrl, download_from_sources
import subprocess

VRM_ADDON_URL = "https://vrm-addon-for-blender.info/releases/VRM_Addon_for_Blender-release.zip"


def _download(url: str, target: Path) -> None:
    result, _ = download_from_sources(
        [DirectUrl(url=url, source="vrm_addon")],
        target,
        timeout=120,
        retries=3,
        pack_id="vrm_addon",
        stage="install_vrm_addon",
    )
    if not result.ok:
        raise RuntimeError(result.error or f"download failed for {url}")


def ensure_vrm_addon_ready(blender_exe: Path, assets_root: Path, sample_vrm: Path | None = None) -> Path:
    marker = assets_root / "addons" / "vrm_addon_ready.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    if marker.exists():
        payload = json.loads(marker.read_text(encoding="utf-8"))
        if payload.get("ok"):
            return marker

    addon_zip = assets_root / "addons" / "VRM_Addon_for_Blender-release.zip"
    if not addon_zip.exists() or addon_zip.stat().st_size < 1024:
        _download(VRM_ADDON_URL, addon_zip)

    result_json = assets_root / "addons" / "vrm_addon_probe.json"
    result_json.parent.mkdir(parents=True, exist_ok=True)
    sample_vrm_value = str(sample_vrm) if sample_vrm else ""
    script = f"""
import bpy, json
addon_zip = r'''{str(addon_zip)}'''
result_path = r'''{str(result_json)}'''
sample_vrm = r'''{sample_vrm_value}''' if {bool(sample_vrm)} else None
bpy.ops.preferences.addon_install(filepath=addon_zip, overwrite=False)
modules = ['VRM_Addon_for_Blender', 'io_scene_vrm']
for mod in bpy.path.module_names(bpy.utils.user_resource('SCRIPTS', 'addons')):
    name = mod.__name__
    if 'vrm' in name.lower() and name not in modules:
        modules.append(name)
enabled = ''
err = ''
for module in modules:
    try:
        bpy.ops.preferences.addon_enable(module=module)
        enabled = module
        break
    except Exception as exc:
        err = str(exc)
bpy.ops.wm.save_userpref()
has_vrm = hasattr(bpy.ops.import_scene, 'vrm')
json.dump({{'ok': bool(enabled) and bool(has_vrm), 'enabled_module': enabled, 'has_vrm': has_vrm, 'error': err, 'modules': modules}}, open(result_path, 'w', encoding='utf-8'))
"""

    cmd = [str(blender_exe), "--background", "--factory-startup", "--python-expr", f"exec({script!r})"]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to install VRM addon: {proc.stderr[-700:]}")
    if not result_json.exists():
        raise RuntimeError("VRM addon install probe did not produce result file")
    probe = json.loads(result_json.read_text(encoding="utf-8"))
    if not probe.get("ok"):
        raise RuntimeError(f"VRM addon install failed: {probe}")

    marker.write_text(
        json.dumps(
            {
                "ok": True,
                "addon_url": VRM_ADDON_URL,
                "sample_vrm": str(sample_vrm.resolve()) if sample_vrm else None,
                "validated_at": datetime.now(timezone.utc).isoformat(),
                "probe": probe,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return marker
