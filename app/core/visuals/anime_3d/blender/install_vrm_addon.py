from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from app.core.net.downloads import DirectUrl, download_from_sources
import subprocess
import tempfile

VRM_ADDON_URL = "https://github.com/saturday06/VRM-Addon-for-Blender/releases/latest/download/VRM_Addon_for_Blender-release.zip"


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


def ensure_vrm_addon_ready(blender_exe: Path, cache_root: Path, sample_vrm: Path) -> Path:
    marker = cache_root / "anime_3d" / "vrm_addon_ready.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    if marker.exists():
        payload = json.loads(marker.read_text(encoding="utf-8"))
        if payload.get("ok") and Path(payload.get("sample_vrm", "")).exists():
            return marker

    addon_zip = cache_root / "anime_3d" / "vrm_addon.zip"
    if not addon_zip.exists() or addon_zip.stat().st_size < 1024:
        _download(VRM_ADDON_URL, addon_zip)

    with tempfile.TemporaryDirectory(prefix="moneyos_vrm_addon_") as temp_dir:
        temp_path = Path(temp_dir)
        result_json = temp_path / "vrm_addon_result.json"
        script_path = temp_path / "install_and_probe_vrm.py"
        script_path.write_text(
            (
                "import bpy, json\n"
                f"addon_zip = r'{str(addon_zip)}'\n"
                f"sample_vrm = r'{str(sample_vrm)}'\n"
                f"result_path = r'{str(result_json)}'\n"
                "bpy.ops.preferences.addon_install(filepath=addon_zip, overwrite=False)\n"
                "for module in ('VRM_Addon_for_Blender-release', 'VRM_Addon_for_Blender', 'io_scene_vrm'):\n"
                "    try:\n"
                "        bpy.ops.preferences.addon_enable(module=module)\n"
                "    except Exception:\n"
                "        pass\n"
                "bpy.ops.wm.save_userpref()\n"
                "before=set(o.name for o in bpy.context.scene.objects)\n"
                "ok=False\n"
                "msg=''\n"
                "try:\n"
                "    bpy.ops.import_scene.vrm(filepath=sample_vrm)\n"
                "except Exception as exc:\n"
                "    msg=str(exc)\n"
                "after=[o for o in bpy.context.scene.objects if o.name not in before]\n"
                "has_armature=any(o.type=='ARMATURE' for o in after)\n"
                "has_mesh=any(o.type=='MESH' for o in after)\n"
                "ok=has_armature and has_mesh\n"
                "json.dump({'ok':ok,'has_armature':has_armature,'has_mesh':has_mesh,'error':msg}, open(result_path,'w',encoding='utf-8'))\n"
            ),
            encoding="utf-8",
        )
        cmd = [str(blender_exe), "--background", "--factory-startup", "--python", str(script_path)]
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"Failed to install VRM addon: {proc.stderr[-500:]}")
        if not result_json.exists():
            raise RuntimeError("VRM addon dry-run import did not produce result file")
        probe = json.loads(result_json.read_text(encoding="utf-8"))
        if not probe.get("ok"):
            raise RuntimeError(f"VRM addon dry-run failed: {probe}")

    marker.write_text(
        json.dumps(
            {
                "ok": True,
                "addon_url": VRM_ADDON_URL,
                "sample_vrm": str(sample_vrm.resolve()),
                "validated_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return marker
