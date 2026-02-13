from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import sys
import wave
from pathlib import Path

import bpy
from mathutils import Vector

from app.core.visuals.anime_3d.asset_scan import (
    ASSET_DIR_ALIASES,
    character_sort_key,
    discover_assets,
    resolve_asset_dir,
    resolve_asset_dirs,
    select_animation_assets,
    select_character_assets,
    select_env_blend,
    strict_assets_error,
)


PHASE3_DEBUG = os.getenv("MONEYOS_DEBUG_PHASE3", "0") == "1" or os.getenv("MONEYOS_PHASE3_DEBUG", "0") == "1"
PHASE3_LOGGER = logging.getLogger("moneyos.phase3")
if not PHASE3_LOGGER.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    PHASE3_LOGGER.addHandler(_handler)
PHASE3_LOGGER.setLevel(logging.DEBUG if PHASE3_DEBUG else logging.INFO)
PHASE3_LOGGER.propagate = False


def _phase3_log(message: str) -> None:
    if not PHASE3_DEBUG and not ("ENTER" in message or "EXIT" in message):
        return
    PHASE3_LOGGER.info(message)


def _phase3_validate_materials(scene: bpy.types.Scene) -> list[dict[str, object]]:
    issues: list[dict[str, object]] = []
    for obj in scene.objects:
        if obj.type != "MESH":
            continue
        if not obj.material_slots:
            issues.append({"object": obj.name, "reason": "no_material_slots"})
            continue
        for slot in obj.material_slots:
            material = slot.material
            if material is None:
                issues.append({"object": obj.name, "reason": "empty_material_slot"})
                continue
            if not material.node_tree:
                issues.append({"object": obj.name, "material": material.name, "reason": "missing_node_tree"})
                continue
            has_output = any(node.type == "OUTPUT_MATERIAL" for node in material.node_tree.nodes)
            if not has_output:
                issues.append({"object": obj.name, "material": material.name, "reason": "missing_material_output"})
    return issues


def _phase3_character_style_check(scene: bpy.types.Scene) -> dict[str, object]:
    armatures = [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]
    candidates = [
        obj
        for obj in bpy.data.objects
        if obj.type == "MESH"
        and (
            "hero" in obj.name.lower()
            or "character" in obj.name.lower()
            or (obj.parent is not None and obj.parent.type == "ARMATURE")
        )
    ]
    material_count = 0
    toon_hits = 0
    for obj in candidates:
        for slot in obj.material_slots:
            mat = slot.material
            if mat is None or mat.node_tree is None:
                continue
            material_count += 1
            for node in mat.node_tree.nodes:
                if node.type == "VALTORGB":
                    toon_hits += 1
                if node.type == "GROUP" and getattr(getattr(node, "node_tree", None), "name", "").lower().find("toon") >= 0:
                    toon_hits += 1
                if node.type == "GROUP" and getattr(getattr(node, "node_tree", None), "name", "").lower().find("anime") >= 0:
                    toon_hits += 1
    passed = bool(armatures) and bool(candidates) and material_count > 0 and toon_hits > 0
    result = {
        "armature_count": len(armatures),
        "mesh_count": len(candidates),
        "materials_detected": material_count,
        "toon_nodegroups_detected": toon_hits,
        "passed": passed,
    }
    if passed:
        _phase3_log(
            "PHASE3_CHARACTER_STYLE_CHECK_PASS "
            f"armature_count={len(armatures)} mesh_count={len(candidates)} "
            f"materials_detected={material_count} toon_nodegroups_detected={toon_hits}"
        )
    else:
        _phase3_log(
            "PHASE3_CHARACTER_STYLE_CHECK_FAIL "
            f"armature_count={len(armatures)} mesh_count={len(candidates)} "
            f"materials_detected={material_count} nodegroups_detected={toon_hits}"
        )
    return result


SMOOTHING_PRESET_DEFAULTS: dict[str, tuple[int, int]] = {
    "POWER_LOW_ANGLE": (8, 8),
    "ORBIT_SNAP": (4, 4),
    "AGGRESSIVE_PUSH_IN": (6, 6),
    "EXTREME_CLOSEUP": (6, 6),
    "STATIC_IMPACT_HOLD": (8, 8),
}


ANIME_TEXTURE_PATTERNS: dict[str, tuple[str, ...]] = {
    "base": ("*_base.png", "*albedo*.png", "*diffuse*.png"),
    "normal": ("*_normal.png", "*normal*.png"),
    "spec": ("*_spec.png", "*spec*.png", "*rough*.png"),
    "detail": ("*_detail.png", "*stripe*.png", "*fold*.png", "*highlight*.png"),
}


def _parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-preset", default="fast_proof")
    parser.add_argument("--engine", default="eevee")
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--audio", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--assets-dir", required=True)
    parser.add_argument("--asset-mode", default="auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fingerprint", default="")
    parser.add_argument("--strict-assets", type=int, default=1)
    parser.add_argument("--environment", default="room")
    parser.add_argument("--beat-plan", default="")
    parser.add_argument("--character-asset", default="")
    parser.add_argument("--character-variation", default="")
    parser.add_argument("--mode", default="default")
    parser.add_argument("--style-preset", default="default")
    parser.add_argument("--outline-mode", default="freestyle")
    parser.add_argument("--postfx", default="on")
    parser.add_argument("--quality", default="balanced")
    parser.add_argument("--res", default="1920x1080")
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--vfx-emission-strength", type=float, default=50.0)
    parser.add_argument("--vfx-scale", type=float, default=1.0)
    parser.add_argument("--vfx-screen-coverage", type=float, default=0.35)
    parser.add_argument("--fast-proof", action="store_true")
    parser.add_argument("--proof-seconds", type=float, default=15.0)
    parser.add_argument("--phase15-samples", type=int, default=128)
    parser.add_argument("--phase15-bounces", type=int, default=6)
    parser.add_argument("--phase15-res", default="1920x1080")
    parser.add_argument("--phase15-tile", type=int, default=256)
    parser.add_argument("--force-procedural-humanoid", type=int, default=0)
    return parser.parse_args(argv)


def _clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in bpy.data.meshes:
        bpy.data.meshes.remove(block)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _derive_seed(output_path: Path, seed_value: int | None) -> int:
    if seed_value is not None:
        return int(seed_value)
    digest = hashlib.sha256(str(output_path.parent).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _seed_randomness(seed_value: int) -> None:
    random.seed(seed_value)
    try:
        import numpy as np  # noqa: WPS433
    except Exception:  # noqa: BLE001
        np = None
    if np is not None:
        np.random.seed(seed_value)
    try:
        scene = bpy.context.scene
        if hasattr(scene, "seed"):
            scene.seed = seed_value
        if hasattr(scene, "cycles") and hasattr(scene.cycles, "seed"):
            scene.cycles.seed = seed_value
    except Exception:  # noqa: BLE001
        pass


def _configure_cycles_gpu_optix(scene: bpy.types.Scene, args: argparse.Namespace) -> dict[str, object]:
    requested_gpu = str(args.gpu) == "1" and (args.engine or "").lower() == "cycles"
    compute_device_type = "NONE"
    devices_info: list[dict[str, object]] = []
    chosen_device = "CPU"
    if requested_gpu:
        try:
            prefs = bpy.context.preferences
            cycles_prefs = prefs.addons["cycles"].preferences
            cycles_prefs.get_devices()
            if any(device.type == "OPTIX" for device in cycles_prefs.devices):
                compute_device_type = "OPTIX"
            elif any(device.type == "CUDA" for device in cycles_prefs.devices):
                compute_device_type = "CUDA"
            cycles_prefs.compute_device_type = compute_device_type
            for device in cycles_prefs.devices:
                device.use = device.type != "CPU"
                devices_info.append(
                    {"name": device.name, "type": device.type, "use": bool(device.use)}
                )
            if compute_device_type in {"OPTIX", "CUDA"} and any(
                device["use"] for device in devices_info if device["type"] != "CPU"
            ):
                scene.cycles.device = "GPU"
                chosen_device = "GPU"
            else:
                scene.cycles.device = "CPU"
                chosen_device = "CPU"
        except Exception:  # noqa: BLE001
            scene.cycles.device = "CPU"
            chosen_device = "CPU"
            compute_device_type = "NONE"
    print(
        "[ANIME3D_GPU] "
        f"engine={args.engine} requested_gpu={int(requested_gpu)} "
        f"compute_device_type={compute_device_type} devices={[(d['name'], d['type'], d['use']) for d in devices_info]} "
        f"chosen_device={chosen_device}"
    )
    if requested_gpu:
        scene.render.engine = "CYCLES"
    quality = (args.quality or "balanced").lower()
    samples = args.phase15_samples
    if quality == "fast":
        samples = 64
    elif quality == "max":
        samples = 128
    elif quality == "balanced":
        samples = 96
    samples = max(64, min(int(samples), 256))
    _safe_set(scene.cycles, "samples", samples)
    _safe_set(scene.cycles, "use_adaptive_sampling", True)
    bounces = max(4, min(int(args.phase15_bounces), 8))
    _safe_set(scene.cycles, "max_bounces", bounces)
    _safe_set(scene.cycles, "diffuse_bounces", 2)
    _safe_set(scene.cycles, "glossy_bounces", 2)
    _safe_set(scene.cycles, "transparent_max_bounces", 2)
    _safe_set(scene.cycles, "caustics_reflective", False)
    _safe_set(scene.cycles, "caustics_refractive", False)
    _safe_set(scene.cycles, "use_denoising", True)
    _safe_set(scene.cycles, "denoiser", "OPTIX" if compute_device_type == "OPTIX" else "OPENIMAGEDENOISE")
    _safe_set(scene.view_settings, "view_transform", "Filmic")
    _safe_set(scene.view_settings, "look", "Medium High Contrast")
    denoiser = getattr(scene.cycles, "denoiser", None)
    print(
        "[ANIME3D_RENDER] "
        f"engine=CYCLES backend={compute_device_type} devices={devices_info} samples={samples} "
        f"bounces={bounces} denoise={denoiser}"
    )
    return {
        "requested": requested_gpu,
        "compute_device_type": compute_device_type,
        "devices": devices_info,
        "scene_device": chosen_device,
    }


def _apply_resolution(scene: bpy.types.Scene, args: argparse.Namespace) -> None:
    width = 1920
    height = 1080
    try:
        width_str, height_str = args.res.lower().split("x", 1)
        width = int(width_str)
        height = int(height_str)
    except ValueError:
        width = 1920
        height = 1080
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    print(f"[ANIME3D_RENDER] FINAL_RES={width}x{height}")


def _get_subject_object(scene: bpy.types.Scene) -> bpy.types.Object | None:
    for obj in scene.objects:
        if obj.type == "MESH" and obj.get("mo_role") == "subject":
            return obj
    for obj in scene.objects:
        if obj.type == "MESH":
            return obj
    return None


def _setup_anime_lighting(scene: bpy.types.Scene, subject_obj: bpy.types.Object | None) -> None:
    collection = bpy.data.collections.get("ANIME_LIGHTS")
    if collection is None:
        collection = bpy.data.collections.new("ANIME_LIGHTS")
        scene.collection.children.link(collection)
    for obj in list(collection.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    subject_location = subject_obj.location if subject_obj else Vector((0, 0, 1))
    key = bpy.data.lights.new(name="KeyLight", type="AREA")
    key.energy = 1800
    key.size = 4.0
    key_obj = bpy.data.objects.new(name="KeyLight", object_data=key)
    key_obj.location = subject_location + Vector((2.5, -2.0, 3.0))
    key_obj.rotation_euler = (math.radians(60), 0, math.radians(40))
    fill = bpy.data.lights.new(name="FillLight", type="AREA")
    fill.energy = 450
    fill.size = 3.0
    fill_obj = bpy.data.objects.new(name="FillLight", object_data=fill)
    fill_obj.location = subject_location + Vector((-3.0, -1.0, 2.0))
    fill_obj.rotation_euler = (math.radians(55), 0, math.radians(-35))
    rim = bpy.data.lights.new(name="RimLight", type="AREA")
    rim.energy = 800
    rim.size = 2.5
    rim.color = (0.7, 0.8, 1.0)
    rim_obj = bpy.data.objects.new(name="RimLight", object_data=rim)
    rim_obj.location = subject_location + Vector((0.0, 3.0, 2.5))
    rim_obj.rotation_euler = (math.radians(120), 0, 0)
    for obj in (key_obj, fill_obj, rim_obj):
        collection.objects.link(obj)
    world = scene.world
    if world:
        world.use_nodes = True
        node_tree = world.node_tree
        if node_tree and "Background" in node_tree.nodes:
            node_tree.nodes["Background"].inputs[1].default_value = 0.1
    world_strength = 0.1
    if world:
        world_strength = node_tree.nodes["Background"].inputs[1].default_value
    print(f"[ANIME3D_LIGHTS] key={key.energy} fill={fill.energy} rim={rim.energy} world={world_strength}")


def _safe_look_at(camera: bpy.types.Object, target: Vector) -> None:
    direction = target - camera.location
    if direction.length <= 1e-6:
        return
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _collect_lights(scene: bpy.types.Scene) -> dict[str, bpy.types.Object]:
    lights: dict[str, bpy.types.Object] = {}
    for obj in scene.objects:
        if obj.type != "LIGHT":
            continue
        name = obj.name.lower()
        if "key" in name and "key" not in lights:
            lights["key"] = obj
        elif "fill" in name and "fill" not in lights:
            lights["fill"] = obj
        elif "rim" in name and "rim" not in lights:
            lights["rim"] = obj
    return lights


def _set_world_strength(scene: bpy.types.Scene, strength: float) -> None:
    world = scene.world
    if not world:
        return
    world.use_nodes = True
    node_tree = world.node_tree
    if node_tree and "Background" in node_tree.nodes:
        node_tree.nodes["Background"].inputs[1].default_value = float(strength)


def _set_principled_input(
    principled: bpy.types.Node,
    names: list[str],
    value: object,
    label: str,
) -> bool:
    for name in names:
        socket = principled.inputs.get(name)
        if socket is not None:
            socket.default_value = value
            return True
    print(f"[MATERIALS] missing Principled input for {label}: {names}")
    if os.getenv("MONEYOS_ANIME3D_DEBUG") == "1":
        available = [socket.name for socket in principled.inputs]
        print(f"[MATERIALS] available Principled inputs: {available}")
    return False


def setup_anime_materials(obj_or_collection: object, preset: str = "default") -> None:
    if isinstance(obj_or_collection, bpy.types.Collection):
        objects = list(obj_or_collection.all_objects)
    elif isinstance(obj_or_collection, bpy.types.Object):
        objects = [obj_or_collection]
    else:
        return
    for obj in objects:
        if obj.type != "MESH":
            continue
        for slot in obj.material_slots:
            mat = slot.material
            if mat is None:
                continue
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            principled = nodes.get("Principled BSDF")
            if principled:
                principled.inputs["Roughness"].default_value = 0.45
                _set_principled_input(
                    principled,
                    ["Specular IOR Level", "Specular"],
                    0.25,
                    "specular",
                )
                principled.inputs["Emission Strength"].default_value = 0.03
            name_lower = mat.name.lower()
            if any(tag in name_lower for tag in ("eye", "iris", "pupil")) and principled:
                principled.inputs["Roughness"].default_value = 0.2
                _set_principled_input(
                    principled,
                    ["Specular IOR Level", "Specular"],
                    0.5,
                    "specular_eye",
                )
                principled.inputs["Emission Strength"].default_value = 0.05
    print("[MATERIALS] anime materials applied")


def setup_anime_camera_motion(
    camera: bpy.types.Object,
    subject: bpy.types.Object | None,
    mode: str,
    duration_frames: int,
    rng_seed: int,
) -> None:
    if subject:
        constraint = camera.constraints.new(type="TRACK_TO")
        constraint.target = subject
        constraint.track_axis = "TRACK_NEGATIVE_Z"
        constraint.up_axis = "UP_Y"
    random.seed(rng_seed)
    start = camera.location.copy()
    end = camera.location.copy()
    if mode == "push_in":
        end.y += 0.4
        end.z += 0.15
    elif mode == "orbit":
        end.x += 0.4
        end.y += 0.2
    elif mode == "handheld":
        end.x += 0.1
        end.y += 0.1
        end.z += 0.05
    camera.location = start
    camera.keyframe_insert(data_path="location", frame=1)
    camera.location = end
    camera.keyframe_insert(data_path="location", frame=duration_frames)
    animation_data = getattr(camera, "animation_data", None)
    action = getattr(animation_data, "action", None)
    fcurves = getattr(action, "fcurves", None)
    if fcurves is None:
        print("[MOTION][WARN] No fcurves available on camera action; skipping curve cleanup.")
    else:
        for fcurve in list(fcurves):
            try:
                keyframe_points = getattr(fcurve, "keyframe_points", [])
                for keyframe in keyframe_points:
                    keyframe.interpolation = "BEZIER"
                    keyframe.handle_left_type = "AUTO_CLAMPED"
                    keyframe.handle_right_type = "AUTO_CLAMPED"
            except Exception as exc:  # noqa: BLE001
                print(f"[MOTION][WARN] Failed to process camera fcurve: {exc}")
    print(f"[CAMERA] mode={mode}")


def apply_anime_animation_polish(
    rig: bpy.types.Object | None,
    duration_frames: int,
    rng_seed: int,
) -> None:
    if rig is None:
        return
    random.seed(rng_seed)
    amplitude = 0.02
    for obj in (rig,):
        obj.location.z += amplitude
        obj.keyframe_insert(data_path="location", frame=1)
        obj.location.z -= amplitude
        obj.keyframe_insert(data_path="location", frame=duration_frames)
        if obj.animation_data:
            action = getattr(getattr(getattr(obj, "animation_data", None), "action", None), "fcurves", None)
            fcurves = action
            if not fcurves:
                print("[MOTION][WARN] No fcurves available on object action; skipping cleanup.")
                return
            for fcurve in list(fcurves):
                for keyframe in fcurve.keyframe_points:
                    keyframe.interpolation = "BEZIER"
                    keyframe.handle_left_type = "AUTO_CLAMPED"
                    keyframe.handle_right_type = "AUTO_CLAMPED"
    print("[ANIM] polish applied")


def _setup_anime_compositor(scene: bpy.types.Scene, args: argparse.Namespace) -> None:
    if args.postfx != "on":
        return
    try:
        scene.use_nodes = True
    except Exception:  # noqa: BLE001
        print("[WARN] Compositor node_tree not available; skipping postfx.")
        return
    tree = getattr(scene, "node_tree", None) or getattr(bpy.context.scene, "node_tree", None)
    if tree is None:
        print("[WARN] Compositor node_tree not available; skipping postfx.")
        return
    tree.nodes.clear()
    render_layers = tree.nodes.new(type="CompositorNodeRLayers")
    color_balance = tree.nodes.new(type="CompositorNodeColorBalance")
    glare = tree.nodes.new(type="CompositorNodeGlare")
    glare.glare_type = "FOG_GLOW"
    glare.threshold = 0.8
    glare.quality = "LOW"
    composite = tree.nodes.new(type="CompositorNodeComposite")
    tree.links.new(render_layers.outputs["Image"], color_balance.inputs["Image"])
    tree.links.new(color_balance.outputs["Image"], glare.inputs["Image"])
    tree.links.new(glare.outputs["Image"], composite.inputs["Image"])
    print("[ANIME3D_COMP] enabled=1")


def _color_from_temperature(temp_k: float) -> tuple[float, float, float]:
    temp = max(1000.0, min(temp_k, 12000.0)) / 100.0
    if temp <= 66:
        red = 1.0
        green = max(0.0, min(1.0, 0.390081578769 * math.log(temp) - 0.631841443788))
        blue = 0.0 if temp <= 19 else max(0.0, min(1.0, 0.54320678911 * math.log(temp - 10) - 1.19625408914))
    else:
        red = max(0.0, min(1.0, 1.29293618606 * ((temp - 60) ** -0.1332047592)))
        green = max(0.0, min(1.0, 1.12989086089 * ((temp - 60) ** -0.0755148492)))
        blue = 1.0
    return red, green, blue




def _resolve_asset_dir(assets_dir: Path, aliases: list[str]) -> Path | None:
    return resolve_asset_dir(assets_dir, aliases)


def _resolve_asset_dirs(assets_dir: Path) -> dict[str, Path | None]:
    return resolve_asset_dirs(assets_dir)


def _character_sort_key(candidate: Path) -> tuple[int, int, str]:
    return character_sort_key(candidate)


def _discover_assets(assets_dir: Path) -> dict[str, list[Path]]:
    inventory, _ = discover_assets(assets_dir)
    return inventory


def _select_env_blend(env_candidates: list[Path], environment: str) -> Path | None:
    return select_env_blend(env_candidates, environment)


def _select_character_assets(char_candidates: list[Path]) -> tuple[Path | None, Path | None]:
    return select_character_assets(char_candidates)


def _select_animation_assets(anim_candidates: list[Path]) -> dict[str, Path | None]:
    return select_animation_assets(anim_candidates)




def _required_assets(assets_dir: Path) -> dict[str, Path]:
    asset_dirs = _resolve_asset_dirs(assets_dir)
    characters_dir = asset_dirs["characters"] or (assets_dir / "characters")
    envs_dir = asset_dirs["envs"] or (assets_dir / "envs")
    anims_dir = asset_dirs["anims"] or (assets_dir / "anims")
    vfx_dir = asset_dirs["vfx"] or (assets_dir / "vfx")
    return {
        "characters/hero.blend": characters_dir / "hero.blend",
        "characters/enemy.blend": characters_dir / "enemy.blend",
        "envs/city.blend": envs_dir / "city.blend",
        "anims/idle.fbx": anims_dir / "idle.fbx",
        "anims/run.fbx": anims_dir / "run.fbx",
        "anims/punch.fbx": anims_dir / "punch.fbx",
        "vfx/explosion.png": vfx_dir / "explosion.png",
        "vfx/energy_arc.png": vfx_dir / "energy_arc.png",
        "vfx/smoke.png": vfx_dir / "smoke.png",
    }


def _find_missing_assets(assets_dir: Path) -> list[str]:
    return [key for key, path in _required_assets(assets_dir).items() if not path.exists()]


def _create_procedural_humanoid(name: str, location: tuple[float, float, float]) -> tuple[bpy.types.Object, int]:
    raise RuntimeError("Procedural primitive humanoids are disabled for anime pipeline.")


def _animate_procedural_humanoid(root: bpy.types.Object, total_frames: int) -> None:
    del root, total_frames


def _build_procedural_scene(
    scene: bpy.types.Scene,
    total_frames: int,
    force_procedural_humanoid: bool,
) -> bool:
    del scene, total_frames, force_procedural_humanoid
    raise RuntimeError("Procedural primitive scenes are disabled for anime pipeline.")


def _find_character_asset(workdir: Path) -> Path | None:
    search_dirs = [workdir, workdir / "assets"]
    for alias in ASSET_DIR_ALIASES["characters"]:
        search_dirs.append(workdir / alias)
    for directory in search_dirs:
        if not directory.exists():
            continue
        for ext in (".glb", ".gltf", ".fbx", ".vrm"):
            matches = sorted(directory.glob(f"*{ext}"))
            if matches:
                return matches[0]
    return None


def _import_character_asset(asset_path: Path) -> tuple[bpy.types.Object | None, str]:
    ext = asset_path.suffix.lower()
    if ext in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(asset_path))
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(asset_path))
    elif ext == ".vrm":
        if hasattr(bpy.ops.import_scene, "vrm"):
            bpy.ops.import_scene.vrm(filepath=str(asset_path))
        else:
            raise RuntimeError("VRM importer add-on is not enabled; run VRM add-on installer first.")
    meshes = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]
    if not meshes:
        meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    subject = meshes[0] if meshes else None
    return subject, ext.lstrip(".")


def _ensure_character(
    scene: bpy.types.Scene,
    args: argparse.Namespace,
    assets_dir: Path,
    seed_value: int,
) -> tuple[bpy.types.Object, str, str]:
    existing_subject = _get_subject_object(scene)
    if existing_subject:
        return existing_subject, "generated", "procedural"
    if args.character_asset:
        asset_path = Path(args.character_asset)
        if asset_path.exists():
            subject, fmt = _import_character_asset(asset_path)
            if subject:
                print(f"[ANIME3D_CHAR] source=provided name={subject.name} format={fmt}")
                return subject, "provided", fmt
    asset_path = None
    chars_dir = assets_dir / "characters"
    if chars_dir.exists():
        discovered: list[Path] = []
        for ext in (".fbx", ".glb", ".gltf", ".blend"):
            discovered.extend(chars_dir.rglob(f"*{ext}"))
        if discovered:
            asset_path = sorted({path for path in discovered}, key=_character_sort_key)[0]
    if asset_path and asset_path.suffix.lower() == ".blend":
        with bpy.data.libraries.load(str(asset_path), link=False) as (data_from, data_to):
            data_to.objects = list(data_from.objects)
        for obj in data_to.objects:
            if obj is None:
                continue
            scene.collection.objects.link(obj)
        subject = next((obj for obj in data_to.objects if obj and obj.type == "MESH"), None)
        if subject:
            print(f"[ANIME3D_CHAR] source=asset_lib name={subject.name} format=blend")
            return subject, "asset_lib", "blend"
    if asset_path:
        subject, fmt = _import_character_asset(asset_path)
        if subject:
            print(f"[ANIME3D_CHAR] source=asset_lib name={subject.name} format={fmt}")
            return subject, "asset_lib", fmt
    raise RuntimeError(
        "Anime character asset was not available. Automatic fallback to primitive humanoids is disabled."
    )


def _apply_outlines(scene: bpy.types.Scene, mode: str) -> None:
    mode = mode.strip().lower()
    thickness = float(os.getenv("MONEYOS_ANIME3D_OUTLINE_THICKNESS", "1.5"))
    thickness = max(0.2, min(thickness, 6.0))
    if mode == "off":
        scene.render.use_freestyle = False
        return
    if mode == "freestyle":
        scene.render.use_freestyle = True
        if hasattr(scene.render, "line_thickness"):
            scene.render.line_thickness = thickness
        return
    scene.render.use_freestyle = False


def _apply_watermark(scene: bpy.types.Scene, label: str) -> None:
    if scene.camera is None:
        return
    bpy.ops.object.text_add(location=(0, 0, 0))
    text_obj = bpy.context.active_object
    text_obj.data.body = label
    text_obj.scale = (0.2, 0.2, 0.2)
    text_obj.location = scene.camera.location + Vector((1.2, 1.2, -0.6))
    print("[ANIME3D_STYLE] key_art_v1 applied")


def _safe_set(obj: object, attr: str, value: object) -> bool:
    if hasattr(obj, attr):
        try:
            setattr(obj, attr, value)
            return True
        except Exception:  # noqa: BLE001
            return False
    return False


def _ensure_material(slot_collection: object, mat: bpy.types.Material) -> None:
    if slot_collection is None:
        return
    mat_name = getattr(mat, "name", None)
    try:
        existing_names = {item.name for item in slot_collection if item is not None}
    except Exception:  # noqa: BLE001
        existing_names = set()
    if mat_name and mat_name in existing_names:
        return
    try:
        slot_collection.append(mat)
    except Exception:  # noqa: BLE001
        try:
            if len(slot_collection) == 0:
                slot_collection.append(mat)
            else:
                slot_collection[0] = mat
        except Exception:  # noqa: BLE001
            return


def _load_rms_envelope(audio_path: Path, fps: int, frame_count: int) -> list[float]:
    if not audio_path.exists():
        return [0.0 for _ in range(frame_count)]
    with wave.open(str(audio_path), "rb") as handle:
        sample_rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
        sample_count = len(frames) // 2
        samples = list(
            int.from_bytes(frames[i : i + 2], byteorder="little", signed=True)
            for i in range(0, len(frames), 2)
        )
    samples_per_frame = max(1, int(sample_rate / fps))
    envelope = []
    for index in range(frame_count):
        start = index * samples_per_frame
        end = min(start + samples_per_frame, sample_count)
        if start >= end:
            envelope.append(0.0)
            continue
        segment = samples[start:end]
        rms = math.sqrt(sum(value * value for value in segment) / len(segment))
        envelope.append(min(rms / 20000.0, 1.0))
    return envelope


def _append_collections(blend_path: Path) -> list[bpy.types.Collection]:
    collections: list[bpy.types.Collection] = []
    with bpy.data.libraries.load(str(blend_path), link=False) as (data_from, data_to):
        data_to.collections = list(data_from.collections)
    for collection in data_to.collections:
        if collection and collection.name not in bpy.context.scene.collection.children:
            bpy.context.scene.collection.children.link(collection)
            collections.append(collection)
    return collections


def _find_armature(collections: list[bpy.types.Collection]) -> bpy.types.Object | None:
    for collection in collections:
        for obj in collection.all_objects:
            if obj.type == "ARMATURE":
                return obj
    return None


def _ensure_toon_node_group() -> bpy.types.NodeTree:
    group_name = "MO_ToonRamp"
    existing = bpy.data.node_groups.get(group_name)
    if existing:
        return existing
    group = bpy.data.node_groups.new(group_name, "ShaderNodeTree")
    group.interface.new_socket(name="Color", in_out="INPUT", socket_type="NodeSocketColor")
    group.interface.new_socket(name="Shaded Color", in_out="OUTPUT", socket_type="NodeSocketColor")
    group.interface.new_socket(name="Normal", in_out="INPUT", socket_type="NodeSocketVector")
    group.interface.new_socket(name="Light", in_out="INPUT", socket_type="NodeSocketVector")
    group.interface.new_socket(name="Hardness", in_out="INPUT", socket_type="NodeSocketFloat")
    nodes = group.nodes
    links = group.links
    input_node = nodes.new(type="NodeGroupInput")
    output_node = nodes.new(type="NodeGroupOutput")
    dot = nodes.new(type="ShaderNodeVectorMath")
    dot.operation = "DOT_PRODUCT"
    multiply = nodes.new(type="ShaderNodeMath")
    multiply.operation = "MULTIPLY"
    add = nodes.new(type="ShaderNodeMath")
    add.operation = "ADD"
    ramp = nodes.new(type="ShaderNodeValToRGB")
    ramp.color_ramp.interpolation = "CONSTANT"
    ramp.color_ramp.elements[0].position = 0.38
    ramp.color_ramp.elements[0].color = (0.58, 0.58, 0.58, 1.0)
    ramp.color_ramp.elements[1].position = 0.68
    ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    mult_color = nodes.new(type="ShaderNodeMixRGB")
    mult_color.blend_type = "MULTIPLY"
    mult_color.inputs[0].default_value = 1.0
    links.new(input_node.outputs["Normal"], dot.inputs[0])
    links.new(input_node.outputs["Light"], dot.inputs[1])
    links.new(dot.outputs[0], multiply.inputs[0])
    links.new(input_node.outputs["Hardness"], multiply.inputs[1])
    add.inputs[1].default_value = 0.5
    links.new(multiply.outputs[0], add.inputs[0])
    links.new(add.outputs[0], ramp.inputs["Fac"])
    links.new(input_node.outputs["Color"], mult_color.inputs[1])
    links.new(ramp.outputs["Color"], mult_color.inputs[2])
    links.new(mult_color.outputs["Color"], output_node.inputs["Shaded Color"])
    return group


def _ensure_rim_node_group() -> bpy.types.NodeTree:
    group_name = "MO_RimBoost"
    existing = bpy.data.node_groups.get(group_name)
    if existing:
        return existing
    group = bpy.data.node_groups.new(group_name, "ShaderNodeTree")
    group.interface.new_socket(name="Color", in_out="INPUT", socket_type="NodeSocketColor")
    group.interface.new_socket(name="Rim Color", in_out="INPUT", socket_type="NodeSocketColor")
    group.interface.new_socket(name="Boost", in_out="INPUT", socket_type="NodeSocketFloat")
    group.interface.new_socket(name="Final", in_out="OUTPUT", socket_type="NodeSocketColor")
    nodes = group.nodes
    links = group.links
    input_node = nodes.new(type="NodeGroupInput")
    output_node = nodes.new(type="NodeGroupOutput")
    fresnel = nodes.new(type="ShaderNodeFresnel")
    fresnel.inputs["IOR"].default_value = 1.2
    boost_mult = nodes.new(type="ShaderNodeMath")
    boost_mult.operation = "MULTIPLY"
    mix = nodes.new(type="ShaderNodeMixRGB")
    links.new(fresnel.outputs["Fac"], boost_mult.inputs[0])
    links.new(input_node.outputs["Boost"], boost_mult.inputs[1])
    links.new(boost_mult.outputs[0], mix.inputs[0])
    links.new(input_node.outputs["Color"], mix.inputs[1])
    links.new(input_node.outputs["Rim Color"], mix.inputs[2])
    links.new(mix.outputs["Color"], output_node.inputs["Final"])
    return group


def _resolve_texture_bundle(textures_dir: Path, object_name: str) -> dict[str, Path]:
    bundle: dict[str, Path] = {}
    prefix = "hero" if any(tag in object_name.lower() for tag in ("hero", "head", "body", "arm", "leg")) else "env"
    for key, patterns in ANIME_TEXTURE_PATTERNS.items():
        matches: list[Path] = []
        for pattern in patterns:
            matches.extend(sorted(textures_dir.glob(f"{prefix}{pattern[1:]}")))
            matches.extend(sorted(textures_dir.glob(pattern)))
        for match in matches:
            if match.exists():
                bundle[key] = match
                break
    return bundle


def _apply_character_variation(scene: bpy.types.Scene, variation: dict[str, object]) -> None:
    hair_color = tuple(variation.get("hair_color", (0.16, 0.2, 0.42, 1.0)))
    eye_color = tuple(variation.get("eye_color", (0.23, 0.72, 0.52, 1.0)))
    clothing_color = tuple(variation.get("clothing_color", (0.18, 0.21, 0.55, 1.0)))
    skin_shift = float(variation.get("skin_tone_shift", 0.0))
    for obj in scene.objects:
        if obj.type != "MESH":
            continue
        for slot in obj.material_slots:
            mat = slot.material
            if mat is None or not mat.use_nodes or not mat.node_tree:
                continue
            principled = mat.node_tree.nodes.get("Principled BSDF")
            if principled is None:
                continue
            name = f"{obj.name}_{mat.name}".lower()
            if "hair" in name:
                principled.inputs["Base Color"].default_value = hair_color
            elif any(tag in name for tag in ("eye", "iris", "pupil")):
                principled.inputs["Base Color"].default_value = eye_color
            elif any(tag in name for tag in ("cloth", "shirt", "jacket", "skirt", "pant")):
                principled.inputs["Base Color"].default_value = clothing_color
            elif any(tag in name for tag in ("skin", "face", "body", "arm", "leg")):
                base = list(principled.inputs["Base Color"].default_value)
                base[0] = min(1.0, max(0.0, base[0] + skin_shift))
                base[1] = min(1.0, max(0.0, base[1] + (skin_shift * 0.6)))
                base[2] = min(1.0, max(0.0, base[2] + (skin_shift * 0.4)))
                principled.inputs["Base Color"].default_value = tuple(base)


def _apply_toon_material(
    obj: bpy.types.Object,
    outline_material: bpy.types.Material,
    texture_bundle: dict[str, Path] | None = None,
    *,
    hardness: float = 1.35,
    outline_thickness: float = 0.02,
) -> None:
    if obj.type != "MESH":
        return
    material = bpy.data.materials.new(name="ToonMaterial")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new(type="ShaderNodeOutputMaterial")
    links = material.node_tree.links
    principled = nodes.new(type="ShaderNodeBsdfPrincipled")
    principled.inputs["Roughness"].default_value = 0.62
    _set_principled_input(principled, ["Specular IOR Level", "Specular"], 0.08, "anime_visual_spec")
    geometry = nodes.new(type="ShaderNodeNewGeometry")
    normalize = nodes.new(type="ShaderNodeVectorMath")
    normalize.operation = "NORMALIZE"
    normal_map = nodes.new(type="ShaderNodeNormalMap")
    light_vector = nodes.new(type="ShaderNodeCombineXYZ")
    light_vector.inputs[0].default_value = 0.2
    light_vector.inputs[1].default_value = 0.6
    light_vector.inputs[2].default_value = 1.0
    toon_group = nodes.new(type="ShaderNodeGroup")
    toon_group.node_tree = _ensure_toon_node_group()
    toon_group.inputs["Hardness"].default_value = float(hardness)
    rim_group = nodes.new(type="ShaderNodeGroup")
    rim_group.node_tree = _ensure_rim_node_group()
    rim_group.inputs["Boost"].default_value = 0.35
    rim_group.inputs["Rim Color"].default_value = (0.84, 0.9, 1.0, 1.0)
    base_rgb = nodes.new(type="ShaderNodeRGB")
    base_rgb.outputs[0].default_value = (0.72, 0.68, 0.64, 1.0)
    links.new(geometry.outputs["Normal"], normalize.inputs[0])
    links.new(normalize.outputs[0], toon_group.inputs["Normal"])
    links.new(light_vector.outputs["Vector"], toon_group.inputs["Light"])
    links.new(base_rgb.outputs["Color"], toon_group.inputs["Color"])
    links.new(toon_group.outputs["Shaded Color"], rim_group.inputs["Color"])
    links.new(rim_group.outputs["Final"], principled.inputs["Base Color"])
    if texture_bundle:
        base_map = texture_bundle.get("base")
        normal = texture_bundle.get("normal")
        spec = texture_bundle.get("spec")
        detail = texture_bundle.get("detail")
        tex_coord = nodes.new(type="ShaderNodeTexCoord")
        mapping = nodes.new(type="ShaderNodeMapping")
        links.new(tex_coord.outputs["UV"], mapping.inputs["Vector"])
        if base_map and base_map.exists():
            tex_base = nodes.new(type="ShaderNodeTexImage")
            tex_base.image = bpy.data.images.load(str(base_map), check_existing=True)
            tex_base.interpolation = "Closest"
            tex_base.extension = "REPEAT"
            links.new(mapping.outputs["Vector"], tex_base.inputs["Vector"])
            links.new(tex_base.outputs["Color"], toon_group.inputs["Color"])
        if detail and detail.exists():
            tex_detail = nodes.new(type="ShaderNodeTexImage")
            tex_detail.image = bpy.data.images.load(str(detail), check_existing=True)
            tex_detail.interpolation = "Closest"
            detail_mix = nodes.new(type="ShaderNodeMixRGB")
            detail_mix.blend_type = "MULTIPLY"
            detail_mix.inputs[0].default_value = 0.2
            links.new(mapping.outputs["Vector"], tex_detail.inputs["Vector"])
            links.new(toon_group.outputs["Shaded Color"], detail_mix.inputs[1])
            links.new(tex_detail.outputs["Color"], detail_mix.inputs[2])
            links.new(detail_mix.outputs["Color"], rim_group.inputs["Color"])
        if normal and normal.exists():
            tex_normal = nodes.new(type="ShaderNodeTexImage")
            tex_normal.image = bpy.data.images.load(str(normal), check_existing=True)
            tex_normal.colorspace_settings.name = "Non-Color"
            links.new(mapping.outputs["Vector"], tex_normal.inputs["Vector"])
            links.new(tex_normal.outputs["Color"], normal_map.inputs["Color"])
            links.new(normal_map.outputs["Normal"], principled.inputs["Normal"])
        if spec and spec.exists():
            tex_spec = nodes.new(type="ShaderNodeTexImage")
            tex_spec.image = bpy.data.images.load(str(spec), check_existing=True)
            tex_spec.colorspace_settings.name = "Non-Color"
            links.new(mapping.outputs["Vector"], tex_spec.inputs["Vector"])
            _set_principled_input(principled, ["Specular IOR Level", "Specular"], 0.2, "anime_visual_spec_map")
            links.new(tex_spec.outputs["Color"], principled.inputs["Roughness"])
    links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    if obj.data.materials:
        obj.data.materials[0] = material
    else:
        obj.data.materials.append(material)
    _ensure_material(obj.data.materials, outline_material)
    print("[MAT] applied outline material to", obj.name)
    modifier = obj.modifiers.new(name="Outline", type="SOLIDIFY")
    modifier.thickness = max(0.002, float(outline_thickness))
    modifier.use_flip_normals = True
    modifier.material_offset = len(obj.data.materials) - 1


def _create_city_env(outline_material: bpy.types.Material) -> None:
    bpy.ops.mesh.primitive_plane_add(size=60, location=(0, 0, 0))
    ground = bpy.context.active_object
    ground.scale = (1, 1, 1)
    _apply_toon_material(ground, outline_material)
    for x in range(-5, 6):
        for y in range(-3, 6):
            height = 2 + (abs(x) + abs(y)) % 5
            bpy.ops.mesh.primitive_cube_add(size=2, location=(x * 4, y * 4, height / 2))
            building = bpy.context.active_object
            building.scale.z = height
            if (x + y) % 3 == 0:
                building.scale.x *= 0.8
                building.scale.y *= 0.7
            _apply_toon_material(building, outline_material)
    for _ in range(40):
        bpy.ops.mesh.primitive_cube_add(size=0.4, location=(math.sin(_) * 6, math.cos(_) * 6, 0.2))
        _apply_toon_material(bpy.context.active_object, outline_material)


def _create_character(name: str, location: tuple[float, float, float]) -> dict[str, bpy.types.Object]:
    del name, location
    raise RuntimeError("Primitive character template generation is disabled.")


def _create_outline_material() -> bpy.types.Material:
    material = bpy.data.materials.new(name="OutlineMaterial")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new(type="ShaderNodeOutputMaterial")
    emission = nodes.new(type="ShaderNodeEmission")
    emission.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    emission.inputs["Strength"].default_value = 1.0
    material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def _import_action(fbx_path: Path) -> bpy.types.Action | None:
    before = set(bpy.data.actions)
    bpy.ops.import_scene.fbx(filepath=str(fbx_path))
    after = [action for action in bpy.data.actions if action not in before]
    return after[-1] if after else None


def _apply_action(armature: bpy.types.Object, action: bpy.types.Action | None) -> None:
    if armature is None or action is None:
        return
    armature.animation_data_create()
    armature.animation_data.action = action
    fcurves = getattr(action, "fcurves", None) if action else None
    if not fcurves:
        print("[MOTION][WARN] No fcurves available on camera action; skipping curve cleanup.")
        return
    for fcurve in list(fcurves):
        mod = fcurve.modifiers.new(type="CYCLES")
        mod.mode_before = "REPEAT"
        mod.mode_after = "REPEAT"


def _find_mouth_shapekey(mesh_objects: list[bpy.types.Object]) -> tuple[bpy.types.Object, bpy.types.ShapeKey] | None:
    for obj in mesh_objects:
        if not obj.data.shape_keys:
            continue
        key = obj.data.shape_keys.key_blocks.get("mouth_open")
        if key:
            return obj, key
    return None

def _anime_assets_dir(assets_dir: Path) -> Path:
    candidates = [
        assets_dir / "assets3d_anime",
        assets_dir.parent / "assets3d_anime",
        Path(__file__).resolve().parent.parent / "assets3d_anime",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _apply_anime_visual_style(scene: bpy.types.Scene, assets_dir: Path, outline_mode: str) -> dict[str, int]:
    outline_material = _create_outline_material()
    anime_assets = _anime_assets_dir(assets_dir)
    textures_dir = anime_assets / "textures"
    imported_chars = 0
    imported_props = 0
    chars_dir = anime_assets / "characters"
    props_dir = anime_assets / "props"
    if chars_dir.exists() and _get_subject_object(scene) is None:
        for char_model in sorted(chars_dir.glob("*.obj")):
            before = {obj.name_full for obj in scene.objects}
            bpy.ops.wm.obj_import(filepath=str(char_model))
            after = [obj for obj in scene.objects if obj.name_full not in before]
            if any(obj.type == "MESH" for obj in after):
                imported_chars += 1
            for obj in after:
                if obj.type == "MESH":
                    obj["mo_role"] = "subject"
    if props_dir.exists():
        for prop_model in sorted(props_dir.glob("*.obj")):
            before = {obj.name_full for obj in scene.objects}
            bpy.ops.wm.obj_import(filepath=str(prop_model))
            after = [obj for obj in scene.objects if obj.name_full not in before]
            if any(obj.type == "MESH" for obj in after):
                imported_props += 1
    mesh_count = 0
    outline_thickness = float(os.getenv("MONEYOS_ANIME3D_OUTLINE_GEOM", "0.018"))
    for obj in scene.objects:
        if obj.type != "MESH":
            continue
        mesh_count += 1
        texture_bundle = _resolve_texture_bundle(textures_dir, obj.name)
        _apply_toon_material(obj, outline_material, texture_bundle, hardness=1.5, outline_thickness=outline_thickness)
    _apply_outlines(scene, outline_mode)
    print("[STYLE] applied anime toon shaders")
    print(f"[ASSETS] loaded characters={imported_chars} props={imported_props}")
    return {"characters": imported_chars, "props": imported_props, "meshes_styled": mesh_count}


def _create_scene(
    assets_dir: Path,
    asset_mode: str,
    env_blend: Path | None,
    hero_asset: Path | None,
    enemy_asset: Path | None,
    default_env: str,
    style_preset: str = "default",
) -> dict[str, bpy.types.Object | None]:
    scene = bpy.context.scene
    outline_material = _create_outline_material()

    hero_armature = None
    enemy_armature = None
    hero_jaw = None
    hero_body = None
    if str(style_preset).strip().lower() == "anime_visual":
        _build_environment_template(default_env)
    elif asset_mode == "local":
        asset_dirs = _resolve_asset_dirs(assets_dir)
        env_path = env_blend or (
            (asset_dirs["envs"] / "city.blend") if asset_dirs["envs"] else None
        )
        hero_path = hero_asset or (
            (asset_dirs["characters"] / "hero.blend") if asset_dirs["characters"] else None
        )
        enemy_path = enemy_asset or (
            (asset_dirs["characters"] / "enemy.blend") if asset_dirs["characters"] else None
        )
        if env_path and env_path.exists():
            env_collections = _append_collections(env_path)
        else:
            env_collections = []
            _build_environment_template(default_env)
        hero_collections = []
        enemy_collections = []
        if hero_path and hero_path.exists():
            hero_collections = _append_collections(hero_path)
        if enemy_path and enemy_path.exists():
            enemy_collections = _append_collections(enemy_path)

        hero_armature = _find_armature(hero_collections)
        enemy_armature = _find_armature(enemy_collections)

        for collection in hero_collections + enemy_collections:
            for obj in collection.all_objects:
                _apply_toon_material(obj, outline_material)

        if hero_armature:
            hero_armature.location = (0, 0, 0)
            for child in hero_armature.children_recursive:
                if child.type == "MESH":
                    child["mo_role"] = "subject"
        elif hero_path is None or not hero_path.exists():
            raise RuntimeError("Character asset missing; primitive fallback is disabled.")
        if enemy_armature:
            enemy_armature.location = (3, -2, 0)
        elif enemy_path is None or not enemy_path.exists():
            raise RuntimeError("Enemy asset missing; primitive fallback is disabled.")
    else:
        raise RuntimeError("Auto primitive character generation is disabled. Provide an anime character asset.")

    bpy.ops.object.camera_add(location=(4, -6, 2.5), rotation=(math.radians(75), 0, math.radians(35)))
    camera = bpy.context.active_object
    scene.camera = camera
    if camera.data:
        camera.data.dof.use_dof = True
        camera.data.dof.focus_distance = 3.0

    return {
        "hero_armature": hero_armature,
        "enemy_armature": enemy_armature,
        "hero_jaw": hero_jaw,
        "hero_body": hero_body,
        "camera": camera,
    }


def _ensure_world_light(scene: bpy.types.Scene, rng: random.Random) -> tuple[float, float]:
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    world = scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    output = nodes.new(type="ShaderNodeOutputWorld")
    background = nodes.new(type="ShaderNodeBackground")
    strength = 1.7
    background.inputs["Strength"].default_value = strength
    hue_variant = rng.uniform(-0.06, 0.06)
    gradient = nodes.new(type="ShaderNodeTexGradient")
    ramp = nodes.new(type="ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.55 + hue_variant, 0.62, 0.7, 1.0)
    ramp.color_ramp.elements[1].color = (0.15, 0.18 + hue_variant, 0.22, 1.0)
    mapping = nodes.new(type="ShaderNodeMapping")
    tex_coord = nodes.new(type="ShaderNodeTexCoord")
    links.new(tex_coord.outputs.get("Generated"), mapping.inputs.get("Vector"))
    links.new(mapping.outputs.get("Vector"), gradient.inputs.get("Vector"))
    links.new(gradient.outputs.get("Fac"), ramp.inputs.get("Fac"))
    links.new(ramp.outputs.get("Color"), background.inputs.get("Color"))
    links.new(background.outputs.get("Background"), output.inputs.get("Surface"))
    return strength, hue_variant


def _ensure_ground_plane() -> bpy.types.Object:
    bpy.ops.mesh.primitive_plane_add(size=20.0, location=(0.0, 0.0, 0.0))
    plane = bpy.context.active_object
    plane["mo_role"] = "ground"
    material = bpy.data.materials.new(name="GroundPlaneMaterial")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new(type="ShaderNodeOutputMaterial")
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    noise = nodes.new(type="ShaderNodeTexNoise")
    ramp = nodes.new(type="ShaderNodeValToRGB")
    noise.inputs["Scale"].default_value = 6.0
    ramp.color_ramp.elements[0].color = (0.2, 0.22, 0.26, 1.0)
    ramp.color_ramp.elements[1].color = (0.55, 0.58, 0.62, 1.0)
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.7
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    if plane.data.materials:
        plane.data.materials[0] = material
    else:
        plane.data.materials.append(material)
    return plane


def _ensure_light_rig(rng: random.Random | None = None) -> tuple[list[bpy.types.Object], dict[str, object], str]:
    lights: list[bpy.types.Object] = []
    bpy.ops.object.light_add(type="AREA", location=(4.0, -3.5, 4.5))
    key = bpy.context.active_object
    key.data.energy = 3500
    key.data.size = 4.5
    _safe_set(key.data, "use_shadow", False)
    lights.append(key)
    bpy.ops.object.light_add(type="AREA", location=(-4.0, -1.5, 3.0))
    fill = bpy.context.active_object
    fill.data.energy = 900
    fill.data.size = 5.0
    _safe_set(fill.data, "use_shadow", False)
    lights.append(fill)
    bpy.ops.object.light_add(type="SPOT", location=(0.0, 4.0, 4.0))
    rim = bpy.context.active_object
    rim.data.energy = 1400
    rim.data.spot_size = math.radians(55)
    _safe_set(rim.data, "use_shadow", False)
    lights.append(rim)
    key_params = {
        "energy": key.data.energy,
        "color": list(getattr(key.data, "color", (1.0, 1.0, 1.0))),
        "rotation": [key.rotation_euler.x, key.rotation_euler.y, key.rotation_euler.z],
    }
    light_variant = "base"
    if rng:
        jitter = rng.uniform(-0.3, 0.3)
        key.rotation_euler.z += jitter
        key.rotation_euler.x += rng.uniform(-0.1, 0.1)
        key.data.energy *= rng.uniform(0.85, 1.15)
        temp = rng.uniform(3600.0, 6800.0)
        key.data.color = _color_from_temperature(temp)
        key_params = {
            "energy": key.data.energy,
            "color": list(getattr(key.data, "color", (1.0, 1.0, 1.0))),
            "rotation": [key.rotation_euler.x, key.rotation_euler.y, key.rotation_euler.z],
            "temperature": temp,
        }
        light_variant = "jittered"
    return lights, key_params, light_variant


def _ensure_visual_density(scene: bpy.types.Scene, duration_s: float, fps: int) -> None:
    frame_start = 1
    frame_end = max(2, int(math.ceil(duration_s * fps)))
    target = next(
        (obj for obj in scene.objects if obj.type == "MESH" and obj.get("mo_role") == "ground"),
        None,
    )
    if target is None:
        target = next((obj for obj in scene.objects if obj.type == "MESH"), None)
    if target:
        material = bpy.data.materials.new(name="Mo_Density")
        material.use_nodes = True
        nodes = material.node_tree.nodes
        links = material.node_tree.links
        nodes.clear()
        output = nodes.new(type="ShaderNodeOutputMaterial")
        bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
        noise = nodes.new(type="ShaderNodeTexNoise")
        ramp = nodes.new(type="ShaderNodeValToRGB")
        mapping = nodes.new(type="ShaderNodeMapping")
        texcoord = nodes.new(type="ShaderNodeTexCoord")
        noise.inputs["Scale"].default_value = 8.0
        links.new(texcoord.outputs["Object"], mapping.inputs["Vector"])
        links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
        links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
        ramp.color_ramp.elements[0].color = (0.1, 0.12, 0.18, 1.0)
        ramp.color_ramp.elements[1].color = (0.55, 0.62, 0.7, 1.0)
        links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
        bsdf.inputs["Roughness"].default_value = 0.6
        links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
        scene.frame_set(frame_start)
        noise.inputs["Scale"].default_value = 8.0
        noise.inputs["Scale"].keyframe_insert(data_path="default_value")
        scene.frame_set(frame_end)
        noise.inputs["Scale"].default_value = 12.0
        noise.inputs["Scale"].keyframe_insert(data_path="default_value")
        if target.data.materials:
            target.data.materials[0] = material
        else:
            target.data.materials.append(material)
    bpy.ops.object.light_add(type="AREA", location=(2.0, -3.0, 4.0))
    extra_light = bpy.context.active_object
    extra_light.data.energy = 350.0
    scene.frame_set(frame_start)
    extra_light.data.energy = 320.0
    extra_light.data.keyframe_insert(data_path="energy")
    scene.frame_set(frame_end)
    extra_light.data.energy = 380.0
    extra_light.data.keyframe_insert(data_path="energy")
    scene.frame_set(frame_start)
    print("[DENSITY] applied procedural noise material and animated light")


def _subject_meshes() -> list[bpy.types.Object]:
    meshes = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    tagged = [obj for obj in meshes if obj.get("mo_role") == "subject"]
    if tagged:
        return tagged
    candidates = [obj for obj in meshes if obj.get("mo_role") != "ground"]
    if not candidates:
        return []
    largest = max(
        candidates,
        key=lambda obj: max(obj.dimensions.x * obj.dimensions.y * obj.dimensions.z, 0.0),
    )
    return [largest]


def _scene_mesh_bounds(meshes: list[bpy.types.Object]) -> tuple[Vector, Vector] | None:
    if not meshes:
        return None
    min_v = Vector((1e9, 1e9, 1e9))
    max_v = Vector((-1e9, -1e9, -1e9))
    for obj in meshes:
        for corner in obj.bound_box:
            world_corner = obj.matrix_world @ Vector(corner)
            min_v.x = min(min_v.x, world_corner.x)
            min_v.y = min(min_v.y, world_corner.y)
            min_v.z = min(min_v.z, world_corner.z)
            max_v.x = max(max_v.x, world_corner.x)
            max_v.y = max(max_v.y, world_corner.y)
            max_v.z = max(max_v.z, world_corner.z)
    return min_v, max_v


def _ensure_subject_proxy() -> bpy.types.Object:
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.2, location=(0.0, 0.0, 1.2))
    proxy = bpy.context.active_object
    proxy["mo_role"] = "subject"
    material = bpy.data.materials.new(name="ProxyMaterial")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new(type="ShaderNodeOutputMaterial")
    diffuse = nodes.new(type="ShaderNodeBsdfDiffuse")
    diffuse.inputs["Color"].default_value = (0.7, 0.65, 0.6, 1.0)
    material.node_tree.links.new(diffuse.outputs["BSDF"], output.inputs["Surface"])
    if proxy.data.materials:
        proxy.data.materials[0] = material
    else:
        proxy.data.materials.append(material)
    return proxy


def _frame_camera(camera: bpy.types.Object) -> dict[str, object] | None:
    meshes = _subject_meshes()
    if not meshes:
        _ensure_subject_proxy()
        meshes = _subject_meshes()
    bounds = _scene_mesh_bounds(meshes)
    if bounds is None or camera.data is None:
        return None
    min_v, max_v = bounds
    center = (min_v + max_v) * 0.5
    height = max(max_v.z - min_v.z, 0.1)
    camera.data.lens = 40
    fov = camera.data.angle
    target_fill = 0.7
    distance = (height * 0.5) / max(math.tan(fov * 0.5), 0.1)
    distance /= target_fill
    camera.location = center + Vector((0.0, -distance * 1.2, height * 0.35))
    direction = center - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    subject_bbox = {
        "min": [min_v.x, min_v.y, min_v.z],
        "max": [max_v.x, max_v.y, max_v.z],
        "height": height,
    }
    camera_params = {
        "lens": camera.data.lens,
        "location": [camera.location.x, camera.location.y, camera.location.z],
        "rotation": [camera.rotation_euler.x, camera.rotation_euler.y, camera.rotation_euler.z],
        "distance": distance,
    }
    print(f"[PHASE2] subject_bbox={subject_bbox} camera={camera_params}")
    return {"subject_bbox": subject_bbox, "camera_params": camera_params}


def _camera_presets() -> list[dict[str, object]]:
    return [
        {"name": "wide_low", "lens": 28, "offset": Vector((0.0, -1.2, 0.2)), "dof": 4.0},
        {"name": "medium_eye", "lens": 40, "offset": Vector((0.0, -1.0, 0.0)), "dof": 3.0},
        {"name": "close_high", "lens": 55, "offset": Vector((0.0, -0.8, 0.25)), "dof": 2.5},
    ]


def _apply_camera_preset(
    camera: bpy.types.Object,
    preset: dict[str, object],
    rng: random.Random,
) -> dict[str, object]:
    if camera.data is None:
        return {}
    lens = float(preset.get("lens", 40))
    camera.data.lens = lens + rng.uniform(-2.0, 2.0)
    if camera.data.dof:
        camera.data.dof.focus_distance = float(preset.get("dof", 3.0)) + rng.uniform(-0.3, 0.3)
    offset = preset.get("offset", Vector((0.0, 0.0, 0.0)))
    if isinstance(offset, Vector):
        camera.location += Vector(
            (
                offset.x + rng.uniform(-0.15, 0.15),
                offset.y + rng.uniform(-0.2, 0.2),
                offset.z + rng.uniform(-0.1, 0.1),
            )
        )
    camera_params = {
        "preset": preset.get("name"),
        "lens": camera.data.lens,
        "location": [camera.location.x, camera.location.y, camera.location.z],
        "rotation": [camera.rotation_euler.x, camera.rotation_euler.y, camera.rotation_euler.z],
        "focus_distance": camera.data.dof.focus_distance if camera.data.dof else None,
    }
    return camera_params


def _setup_visibility_scene(
    scene: bpy.types.Scene,
    camera: bpy.types.Object | None,
    rng: random.Random,
) -> dict[str, object]:
    world_strength, hue_variant = _ensure_world_light(scene, rng)
    _ensure_ground_plane()
    _, key_light_params, light_variant = _ensure_light_rig(rng)
    _safe_set(scene.view_settings, "exposure", 0.8 + rng.uniform(-0.15, 0.15))
    framing = _frame_camera(camera) if camera else None
    camera_params = framing["camera_params"] if framing else None
    camera_preset_name = None
    camera_variant = None
    if camera:
        preset = rng.choice(_camera_presets())
        camera_preset_name = preset.get("name")
        camera_params = _apply_camera_preset(camera, preset, rng)
        camera_variant = camera_preset_name
    return {
        "world_strength": world_strength,
        "subject_bbox": framing["subject_bbox"] if framing else None,
        "camera_params": camera_params,
        "camera_preset": camera_preset_name,
        "key_light_params": key_light_params,
        "camera_variant": camera_variant,
        "light_variant": light_variant,
        "hue_variant": hue_variant,
    }


def _build_shot_plan(
    beat_plan: list[dict[str, object]],
    fps: int,
    total_frames: int,
    seed_value: int,
) -> list[dict[str, object]]:
    rng = random.Random(seed_value + 7919)
    min_shot = max(1, int(2 * fps))
    max_shot = max(min_shot, int(6 * fps))
    beat_frames: list[int] = []
    for beat in beat_plan:
        start_s = beat.get("start")
        if start_s is None:
            start_s = beat.get("start_seconds", beat.get("t_start", beat.get("time", 0.0)))
        try:
            frame = int(round(float(start_s) * fps)) + 1
        except Exception:  # noqa: BLE001
            continue
        frame = max(2, min(total_frames, frame))
        beat_frames.append(frame)
    beat_frames = sorted(set(beat_frames))
    cuts = [1]
    cursor = 1
    while cursor < total_frames:
        target = cursor + rng.randint(min_shot, max_shot)
        candidates = [frame for frame in beat_frames if cursor + min_shot <= frame <= cursor + max_shot]
        next_cut = min(candidates, key=lambda frame: abs(frame - target)) if candidates else target
        next_cut = max(cursor + min_shot, min(next_cut, cursor + max_shot, total_frames))
        if total_frames - next_cut < min_shot:
            next_cut = total_frames
        if next_cut <= cursor:
            next_cut = min(total_frames, cursor + min_shot)
        cuts.append(next_cut)
        cursor = next_cut
    if cuts[-1] != total_frames:
        cuts.append(total_frames)
    presets = [
        "POWER_LOW_ANGLE",
        "AGGRESSIVE_PUSH_IN",
        "ORBIT_SNAP",
        "EXTREME_CLOSEUP",
        "STATIC_IMPACT_HOLD",
    ]
    impacts = {
        max(2, int(total_frames * 0.4)),
        max(3, int(total_frames * 0.8)),
    }
    shots: list[dict[str, object]] = []
    for idx in range(len(cuts) - 1):
        start_frame = cuts[idx]
        end_frame = cuts[idx + 1]
        if idx < len(cuts) - 2:
            end_frame -= 1
        shot_mid = (start_frame + end_frame) // 2
        impact = any(abs(shot_mid - anchor) <= int(1.5 * fps) for anchor in impacts)
        preset = rng.choice(presets[:-1])
        if impact:
            preset = "STATIC_IMPACT_HOLD" if rng.random() < 0.5 else "AGGRESSIVE_PUSH_IN"
        pre_hold_frames, post_ease_frames = SMOOTHING_PRESET_DEFAULTS.get(preset, (6, 6))
        if (end_frame - start_frame + 1) < int(2 * fps):
            pre_hold_frames = 0
            post_ease_frames = 0

        shot = {
            "shot_index": idx + 1,
            "name": f"SHOT_{idx + 1:02d}",
            "start_frame": start_frame,
            "end_frame": end_frame,
            "camera_preset": preset,
            "motion_preset": preset,
            "lens": rng.randint(28, 40) if preset in {"POWER_LOW_ANGLE", "AGGRESSIVE_PUSH_IN", "ORBIT_SNAP"} else rng.randint(50, 85),
            "distance": round(rng.uniform(2.0, 4.8), 3),
            "angle": round(rng.uniform(-14.0, 18.0), 3),
            "shake": round(rng.uniform(0.0, 0.05), 3),
            "hold_frames": rng.randint(8, 12) if impact else 0,
            "pre_hold_frames": pre_hold_frames,
            "post_ease_frames": post_ease_frames,
            "transition_motion": {
                "bias_strength": 0.08 if preset in {"AGGRESSIVE_PUSH_IN", "ORBIT_SNAP"} else 0.05,
                "next_pull": 0.06,
                "enable_micro_shake": bool(impact or preset == "POWER_LOW_ANGLE"),
                "shake_strength": 0.03 if impact else (0.02 if preset == "POWER_LOW_ANGLE" else 0.0),
            },

            "vfx_preset": "IMPACT_BURST" if impact else "NONE",
            "lighting_preset": rng.choice([
    "RIM_HEAVY",
    "DARK_CONTRAST",
    "DRAMATIC_KEY",
    "SILHOUETTE_BACKLIGHT",
]),
            "lighting_intensity": round(rng.uniform(0.85, 1.35), 2),
            "lighting_color": rng.choice(["#9BB8FF", "#FFD9B3", "#FFFFFF", "#FFB8CC"]),
            "impact": impact,
        }
        if impact and shot["lighting_preset"] == "DARK_CONTRAST":
            shot["lighting_preset"] = "EXPLOSION_FLASH"
        shots.append(shot)
    impact_count = sum(1 for shot in shots if int(shot.get("hold_frames", 0)) > 0)
    if impact_count < 2 and shots:
        for marker in (int(len(shots) * 0.4), int(len(shots) * 0.8)):
            idx = min(len(shots) - 1, max(0, marker))
            shots[idx]["hold_frames"] = max(8, int(shots[idx].get("hold_frames", 0)))
            shots[idx]["vfx_preset"] = "IMPACT_BURST"
            shots[idx]["lighting_preset"] = "EXPLOSION_FLASH"
    return shots


def _apply_lighting_preset(scene: bpy.types.Scene, preset: str) -> None:
    lights = _collect_lights(scene)
    key = lights.get("key")
    fill = lights.get("fill")
    rim = lights.get("rim")
    intensity_mult = 1.0
    color = (1.0, 1.0, 1.0)
    if "|" in preset:
        segments = preset.split("|")
        preset = segments[0]
        for token in segments[1:]:
            if token.startswith("intensity="):
                try:
                    intensity_mult = max(0.2, min(3.0, float(token.split("=", 1)[1])))
                except ValueError:
                    pass
            if token.startswith("color="):
                raw = token.split("=", 1)[1].strip()
                if raw.startswith("#") and len(raw) == 7:
                    color = (
                        int(raw[1:3], 16) / 255.0,
                        int(raw[3:5], 16) / 255.0,
                        int(raw[5:7], 16) / 255.0,
                    )
    if preset == "RIM_HEAVY":
        if key and key.data:
            key.data.energy = 1800 * intensity_mult
        if fill and fill.data:
            fill.data.energy = 250 * intensity_mult
        if rim and rim.data:
            rim.data.energy = 1450 * intensity_mult
            rim.data.color = color
        _set_world_strength(scene, 0.05)
    elif preset == "DRAMATIC_KEY":
        if key and key.data:
            key.data.energy = 2800 * intensity_mult
            key.data.color = color
        if fill and fill.data:
            fill.data.energy = 140 * intensity_mult
        if rim and rim.data:
            rim.data.energy = 1000 * intensity_mult
        _set_world_strength(scene, 0.02)
    elif preset == "SILHOUETTE_BACKLIGHT":
        if key and key.data:
            key.data.energy = 400 * intensity_mult
            key.data.color = (0.85, 0.88, 1.0)
        if fill and fill.data:
            fill.data.energy = 60 * intensity_mult
        if rim and rim.data:
            rim.data.energy = 2600 * intensity_mult
            rim.data.color = color
        _set_world_strength(scene, 0.01)
    elif preset == "DARK_CONTRAST":
        if key and key.data:
            key.data.energy = 2200 * intensity_mult
        if fill and fill.data:
            fill.data.energy = 160 * intensity_mult
        if rim and rim.data:
            rim.data.energy = 900 * intensity_mult
            rim.data.color = color
        _set_world_strength(scene, 0.03)
    elif preset == "EXPLOSION_FLASH":
        if key and key.data:
            key.data.energy = 2600 * intensity_mult
        if fill and fill.data:
            fill.data.energy = 1400 * intensity_mult
        if rim and rim.data:
            rim.data.energy = 1800 * intensity_mult
            rim.data.color = color
        _set_world_strength(scene, 0.35)


def _apply_shot_camera(
    scene: bpy.types.Scene,
    camera: bpy.types.Object,
    subject_obj: bpy.types.Object | None,
    shot: dict[str, object],
    next_shot: dict[str, object] | None = None,
) -> None:
    target = subject_obj.location.copy() if subject_obj else Vector((0.0, 0.0, 1.2))
    lens = float(shot.get("lens", 40))
    if camera.data:
        camera.data.lens = lens
    distance = float(shot.get("distance", 3.0))
    low_angle = math.radians(float(shot.get("angle", 0.0)))
    base_height = max(0.45, target.z + (0.2 if shot.get("camera_preset") == "POWER_LOW_ANGLE" else 0.9))
    start = target + Vector((0.0, -distance, base_height - target.z))
    end = start.copy()
    preset = str(shot.get("camera_preset", "STATIC_IMPACT_HOLD"))
    if preset == "POWER_LOW_ANGLE":
        start.z = target.z - 0.5
        end.z = target.z - 0.35
    elif preset == "AGGRESSIVE_PUSH_IN":
        end.y += distance * 0.55
    elif preset == "ORBIT_SNAP":
        start.x -= 0.8
        end.x += 0.6
        end.y += distance * 0.2
    elif preset == "EXTREME_CLOSEUP":
        start.y += distance * 0.6
        end.y += distance * 0.72
    hold_frames = max(0, int(shot.get("hold_frames", 0)))
    frame_start = int(shot["start_frame"])
    frame_end = int(shot["end_frame"])
    shot_len = max(2, frame_end - frame_start + 1)
    hold_start = max(frame_start + 1, frame_end - hold_frames + 1)
    overshoot = frame_start + max(1, int(shot_len * 0.22))
    settle = frame_start + max(2, int(shot_len * 0.5))
    scene.frame_set(frame_start)
    camera.location = start
    _safe_look_at(camera, target)
    camera.rotation_euler.x += low_angle
    camera.keyframe_insert(data_path="location", frame=frame_start)
    camera.keyframe_insert(data_path="rotation_euler", frame=frame_start)
    scene.frame_set(overshoot)
    camera.location = end + (end - start) * 0.12
    _safe_look_at(camera, target)
    camera.rotation_euler.x += low_angle
    camera.keyframe_insert(data_path="location", frame=overshoot)
    camera.keyframe_insert(data_path="rotation_euler", frame=overshoot)
    scene.frame_set(settle)
    camera.location = end
    _safe_look_at(camera, target)
    camera.rotation_euler.x += low_angle
    camera.keyframe_insert(data_path="location", frame=settle)
    camera.keyframe_insert(data_path="rotation_euler", frame=settle)
    if hold_frames > 0:
        scene.frame_set(hold_start)
        hold_loc = camera.location.copy()
        hold_rot = camera.rotation_euler.copy()
        camera.keyframe_insert(data_path="location", frame=hold_start)
        camera.keyframe_insert(data_path="rotation_euler", frame=hold_start)
        scene.frame_set(frame_end)
        camera.location = hold_loc
        camera.rotation_euler = hold_rot
        camera.keyframe_insert(data_path="location", frame=frame_end)
        camera.keyframe_insert(data_path="rotation_euler", frame=frame_end)
    action = getattr(getattr(camera, "animation_data", None), "action", None)
    if action:
        fcurves = getattr(action, "fcurves", None) if action else None
        if not fcurves:
            print("[MOTION][WARN] No fcurves available on camera action; skipping curve cleanup.")
            return
        for fcurve in list(fcurves):
            for kp in fcurve.keyframe_points:
                if kp.co.x <= overshoot:
                    kp.interpolation = "LINEAR"
                elif kp.co.x <= settle:
                    kp.interpolation = "BEZIER"


def _apply_impact_vfx(scene: bpy.types.Scene, shot: dict[str, object], emission_strength: float) -> None:
    if int(shot.get("hold_frames", 0)) <= 0:
        return
    hold_frames = int(shot.get("hold_frames", 0))
    frame_end = int(shot["end_frame"])
    hold_start = max(int(shot["start_frame"]), frame_end - hold_frames + 1)
    for obj in scene.objects:
        if obj.type == "MESH" and obj.name.lower().startswith("vfx_"):
            scene.frame_set(hold_start)
            obj.hide_render = False
            obj.keyframe_insert(data_path="hide_render", frame=hold_start)
            obj.scale = obj.scale * 1.0
            obj.keyframe_insert(data_path="scale", frame=hold_start)
            scene.frame_set(frame_end)
            obj.scale = obj.scale * 1.2
            obj.keyframe_insert(data_path="scale", frame=frame_end)
            break
    for obj in scene.objects:
        if obj.type != "LIGHT" or not getattr(obj, "data", None):
            continue
        base_energy = float(getattr(obj.data, "energy", 0.0))
        scene.frame_set(hold_start)
        obj.data.energy = base_energy
        obj.data.keyframe_insert(data_path="energy", frame=hold_start)
        scene.frame_set(frame_end)
        obj.data.energy = base_energy + emission_strength * 1.5
        obj.data.keyframe_insert(data_path="energy", frame=frame_end)


def _normalize_character(objects: list[bpy.types.Object]) -> list[bpy.types.Object]:
    meshes = [obj for obj in objects if obj.type == "MESH"]
    if not meshes:
        return []
    bounds = _scene_mesh_bounds(meshes)
    if bounds is None:
        return []
    min_v, max_v = bounds
    height = max(max_v.z - min_v.z, 0.1)
    target_height = 1.7
    scale_factor = target_height / height
    for obj in objects:
        obj.scale = (obj.scale.x * scale_factor, obj.scale.y * scale_factor, obj.scale.z * scale_factor)
    bpy.context.view_layer.update()
    bounds = _scene_mesh_bounds(meshes)
    if bounds is None:
        return []
    min_v, _ = bounds
    z_offset = -min_v.z
    for obj in objects:
        obj.location.z += z_offset
    for obj in meshes:
        obj["mo_role"] = "subject"
    return meshes


def _load_character_asset(assets_dir: Path, character_asset: str, warnings: list[str]) -> list[bpy.types.Object]:
    if not character_asset:
        return []
    asset_path = Path(character_asset)
    if not asset_path.is_file():
        asset_dirs = _resolve_asset_dirs(assets_dir)
        characters_dir = asset_dirs["characters"] or (assets_dir / "characters")
        asset_path = characters_dir / character_asset
    if asset_path.is_dir():
        blend_files = sorted(asset_path.glob("*.blend"))
        if blend_files:
            asset_path = blend_files[0]
    if not asset_path.exists():
        warnings.append("character_asset_missing")
        print(f"[PHASE2] character asset missing: {asset_path}")
        return []
    if asset_path.suffix.lower() == ".blend":
        collections = _append_collections(asset_path)
        objects: list[bpy.types.Object] = []
        for collection in collections:
            objects.extend(list(collection.all_objects))
        return _normalize_character(objects)
    if asset_path.suffix.lower() == ".vrm":
        if not hasattr(bpy.ops.import_scene, "vrm"):
            raise RuntimeError("VRM importer add-on is not enabled")
        bpy.ops.import_scene.vrm(filepath=str(asset_path))
        meshes = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]
        return _normalize_character(meshes)
    warnings.append("character_asset_unsupported")
    print(f"[PHASE2] unsupported character asset: {asset_path}")
    return []


def _build_environment_template(template: str) -> None:
    def apply_env_material(obj: bpy.types.Object) -> None:
        material = bpy.data.materials.new(name="EnvMaterial")
        material.use_nodes = True
        nodes = material.node_tree.nodes
        nodes.clear()
        output = nodes.new(type="ShaderNodeOutputMaterial")
        diffuse = nodes.new(type="ShaderNodeBsdfDiffuse")
        diffuse.inputs["Color"].default_value = (0.4, 0.4, 0.42, 1.0)
        material.node_tree.links.new(diffuse.outputs["BSDF"], output.inputs["Surface"])
        if obj.data.materials:
            obj.data.materials[0] = material
        else:
            obj.data.materials.append(material)

    if template == "street":
        bpy.ops.mesh.primitive_plane_add(size=30.0, location=(0.0, 0.0, 0.0))
        ground = bpy.context.active_object
        ground["mo_role"] = "ground"
        apply_env_material(ground)
        for offset in (-6.0, 6.0):
            bpy.ops.mesh.primitive_plane_add(size=12.0, location=(offset, 6.0, 4.0))
            wall = bpy.context.active_object
            wall.rotation_euler.x = math.radians(90)
            apply_env_material(wall)
    elif template == "studio":
        bpy.ops.mesh.primitive_plane_add(size=20.0, location=(0.0, -4.0, 0.0))
        floor = bpy.context.active_object
        floor["mo_role"] = "ground"
        apply_env_material(floor)
        bpy.ops.mesh.primitive_plane_add(size=20.0, location=(0.0, 6.0, 6.0))
        backdrop = bpy.context.active_object
        backdrop.rotation_euler.x = math.radians(90)
        apply_env_material(backdrop)
    else:
        bpy.ops.mesh.primitive_plane_add(size=14.0, location=(0.0, 0.0, 0.0))
        floor = bpy.context.active_object
        floor["mo_role"] = "ground"
        apply_env_material(floor)
        bpy.ops.mesh.primitive_plane_add(size=14.0, location=(0.0, 7.0, 4.0))
        wall = bpy.context.active_object
        wall.rotation_euler.x = math.radians(90)
        apply_env_material(wall)


def _build_environment_collection(template: str, name: str) -> bpy.types.Collection:
    scene = bpy.context.scene
    collection = bpy.data.collections.new(name)
    scene.collection.children.link(collection)
    before = set(scene.objects)
    _build_environment_template(template)
    after = set(scene.objects)
    new_objects = [obj for obj in after - before if obj.name in bpy.context.scene.objects]
    for obj in new_objects:
        if obj.users_collection:
            for col in list(obj.users_collection):
                col.objects.unlink(obj)
        collection.objects.link(obj)
    return collection


def _set_collection_render_visibility(
    col: bpy.types.Collection,
    visible: bool,
    frame: int,
    keyframe: bool = True,
) -> None:
    hide = not visible
    if hasattr(col, "hide_render"):
        try:
            col.hide_render = hide
        except Exception:  # noqa: BLE001
            pass
    if hasattr(col, "hide_viewport"):
        try:
            col.hide_viewport = hide
        except Exception:  # noqa: BLE001
            pass
    objs = []
    if hasattr(col, "all_objects"):
        objs = list(col.all_objects)
    elif hasattr(col, "objects"):
        objs = list(col.objects)
    for obj in objs:
        try:
            obj.hide_render = hide
            if keyframe:
                obj.keyframe_insert(data_path="hide_render", frame=frame)
        except Exception:  # noqa: BLE001
            pass
        try:
            obj.hide_viewport = hide
            if keyframe:
                obj.keyframe_insert(data_path="hide_viewport", frame=frame)
        except Exception:  # noqa: BLE001
            pass


def _apply_environment_schedule(
    schedule: list[dict[str, object]],
    fps: int,
    default_env: str,
) -> str:
    scene = bpy.context.scene
    if not schedule:
        _build_environment_template(default_env)
        return default_env
    env_names = []
    collections = {}
    for idx, beat in enumerate(schedule):
        env_name = str(beat.get("environment", default_env))
        if env_name in collections:
            continue
        col = _build_environment_collection(env_name, f"ENV_{idx}_{env_name}")
        collections[env_name] = col
        env_names.append(env_name)
    if not env_names:
        _build_environment_template(default_env)
        return default_env
    for beat in schedule:
        env_name = str(beat.get("environment", default_env))
        t0 = float(beat.get("t0", 0))
        t1 = float(beat.get("t1", t0 + 1))
        start_frame = max(1, int(t0 * fps))
        end_frame = max(start_frame + 1, int(t1 * fps))
        for name, col in collections.items():
            scene.frame_set(start_frame)
            _set_collection_render_visibility(col, name == env_name, start_frame)
            scene.frame_set(end_frame)
            _set_collection_render_visibility(col, name == env_name, end_frame)
    return env_names[0]


def _apply_environment_schedule_local(
    schedule: list[dict[str, object]],
    fps: int,
    env_candidates: list[Path],
    default_env: str,
) -> str:
    if not schedule or not env_candidates:
        return default_env
    env_map = {path.stem.lower(): path for path in env_candidates}
    collections: dict[str, list[bpy.types.Collection]] = {}
    for beat in schedule:
        env_name = str(beat.get("environment", default_env)).lower()
        path = env_map.get(env_name) or env_candidates[0]
        if env_name not in collections:
            collections[env_name] = _append_collections(path)
    for beat in schedule:
        env_name = str(beat.get("environment", default_env)).lower()
        t0 = float(beat.get("t0", 0))
        t1 = float(beat.get("t1", t0 + 1))
        start_frame = max(1, int(t0 * fps))
        end_frame = max(start_frame + 1, int(t1 * fps))
        for name, cols in collections.items():
            for col in cols:
                bpy.context.scene.frame_set(start_frame)
                _set_collection_render_visibility(col, name == env_name, start_frame)
                bpy.context.scene.frame_set(end_frame)
                _set_collection_render_visibility(col, name == env_name, end_frame)
    return schedule[0].get("environment", default_env)


def _build_viseme_schedule(text: str, frame_count: int) -> list[float]:
    if not text:
        return [0.0 for _ in range(frame_count)]
    vowels = "aeiou"
    values = []
    for char in text.lower():
        if char in vowels:
            values.append(1.0)
        elif char.isalpha():
            values.append(0.4)
    if not values:
        return [0.0 for _ in range(frame_count)]
    schedule = []
    for idx in range(frame_count):
        value = values[int(idx / frame_count * len(values))]
        schedule.append(value)
    return schedule

def _animate(
    objects: dict[str, bpy.types.Object | None],
    envelope: list[float],
    viseme_schedule: list[float],
    fps: int,
    assets_dir: Path,
    asset_mode: str,
    fast_proof: bool,
    mode: str,
) -> int:
    hero_armature = objects["hero_armature"]
    camera = objects["camera"]
    frame_count = len(envelope)
    mouth_keyframes = 0
    action = None
    if asset_mode == "local":
        asset_dirs = _resolve_asset_dirs(assets_dir)
        anims_dir = asset_dirs["anims"]
        anim_candidates = sorted(anims_dir.glob("*.fbx")) if anims_dir else []
        selection = _select_animation_assets(anim_candidates)
        action_path = selection.get("run")
        if action_path and action_path.exists():
            action = _import_action(action_path)
    _apply_action(hero_armature, action)
    mesh_objects = []
    if hero_armature:
        mesh_objects = [child for child in hero_armature.children_recursive if child.type == "MESH"]
    mouth_key = _find_mouth_shapekey(mesh_objects)
    jaw_bone = None
    if hero_armature:
        for name in ("jaw", "Jaw", "JAW"):
            if name in hero_armature.pose.bones:
                jaw_bone = hero_armature.pose.bones[name]
                break

    if camera and camera.data:
        camera.data.lens = 18
    for frame in range(frame_count):
        bpy.context.scene.frame_set(frame + 1)
        t = frame / fps
        if hero_armature and mode != "static_pose":
            hero_armature.location.x = t * 0.03
            hero_armature.keyframe_insert(data_path="location", index=0)
        elif objects.get("hero_body") and mode != "static_pose":
            hero_body = objects["hero_body"]
            hero_body.rotation_euler.z = math.sin(t * 2.0) * 0.2
            hero_body.keyframe_insert(data_path="rotation_euler", index=2)
        if camera:
            camera.location.x = camera.location.x + math.sin(t * 0.8) * 0.05
            camera.location.y = camera.location.y + math.cos(t * 0.7) * 0.05
            camera.location.z = camera.location.z + math.sin(t * 1.2) * 0.03
            camera.keyframe_insert(data_path="location", index=-1)
        viseme_value = viseme_schedule[frame] if frame < len(viseme_schedule) else envelope[frame]
        if jaw_bone:
            jaw_bone.rotation_euler.x = -max(envelope[frame], viseme_value) * 0.6
            jaw_bone.keyframe_insert(data_path="rotation_euler", index=0)
            if max(envelope[frame], viseme_value) > 0.02:
                mouth_keyframes += 1
        elif objects.get("hero_jaw"):
            jaw_obj = objects["hero_jaw"]
            jaw_obj.rotation_euler.x = -max(envelope[frame], viseme_value) * 0.6
            jaw_obj.keyframe_insert(data_path="rotation_euler", index=0)
            if max(envelope[frame], viseme_value) > 0.02:
                mouth_keyframes += 1
        elif mouth_key:
            _, key = mouth_key
            key.value = min(1.0, max(envelope[frame], viseme_value) * 1.2)
            key.keyframe_insert(data_path="value")
            if max(envelope[frame], viseme_value) > 0.02:
                mouth_keyframes += 1
    return mouth_keyframes


def _ensure_vfx_collection(scene: bpy.types.Scene) -> bpy.types.Collection:
    collection = bpy.data.collections.get("VFX")
    if collection is None:
        collection = bpy.data.collections.new("VFX")
        scene.collection.children.link(collection)
    elif collection.name not in scene.collection.children:
        scene.collection.children.link(collection)
    if hasattr(collection, "hide_viewport"):
        collection.hide_viewport = False
    if hasattr(collection, "hide_render"):
        collection.hide_render = False
    return collection


def _apply_vfx_material(
    plane: bpy.types.Object,
    image_path: Path,
    emission_strength: float,
    warnings: list[str],
) -> None:
    material = bpy.data.materials.new(name="VFX_MAT")
    material.use_nodes = True
    node_tree = material.node_tree
    nodes = node_tree.nodes
    links = node_tree.links
    nodes.clear()

    output = nodes.new(type="ShaderNodeOutputMaterial")
    tex = nodes.new(type="ShaderNodeTexImage")
    tex.image = bpy.data.images.load(str(image_path))
    tex.image.colorspace_settings.name = "sRGB"

    emission = nodes.new(type="ShaderNodeEmission")
    emission.inputs["Strength"].default_value = float(emission_strength)
    transparent = nodes.new(type="ShaderNodeBsdfTransparent")
    mix = nodes.new(type="ShaderNodeMixShader")

    links.new(tex.outputs["Color"], emission.inputs["Color"])
    if "Alpha" in tex.outputs:
        links.new(tex.outputs["Alpha"], mix.inputs["Fac"])
    else:
        mix.inputs["Fac"].default_value = 1.0
    links.new(transparent.outputs["BSDF"], mix.inputs[1])
    links.new(emission.outputs["Emission"], mix.inputs[2])
    links.new(mix.outputs["Shader"], output.inputs["Surface"])

    material.blend_method = "BLEND"
    if hasattr(material, "shadow_method"):
        material.shadow_method = "NONE"
    else:
        print("[VFX] shadow_method missing; skipping")
        warnings.append("vfx_shadow_method_missing")
    material.use_backface_culling = False
    if hasattr(material, "alpha_threshold"):
        material.alpha_threshold = 0.0
    if plane.data.materials:
        plane.data.materials[0] = material
    else:
        plane.data.materials.append(material)
    print("[VFX] emissive material applied:", image_path.name, "strength", emission_strength)


def _ensure_minimum_motion(
    objects: dict[str, bpy.types.Object | None],
    scene: bpy.types.Scene,
    duration_s: float,
    fps: int,
) -> dict[str, bool]:
    frame_start = 1
    frame_end = max(2, int(math.ceil(duration_s * fps)))
    camera_motion = False
    character_motion = False
    object_motion = False
    light_motion = False
    camera = objects.get("camera")
    if camera:
        scene.frame_set(frame_start)
        camera.location.x += 0.15
        camera.location.y += 0.1
        camera.keyframe_insert(data_path="location")
        scene.frame_set(frame_end)
        camera.location.x -= 0.3
        camera.location.y -= 0.2
        camera.keyframe_insert(data_path="location")
        camera_motion = True
    hero = objects.get("hero_armature") or objects.get("hero_body")
    if hero:
        scene.frame_set(frame_start)
        hero.rotation_euler.z += 0.1
        hero.keyframe_insert(data_path="rotation_euler", index=2)
        scene.frame_set(frame_end)
        hero.rotation_euler.z -= 0.2
        hero.keyframe_insert(data_path="rotation_euler", index=2)
        character_motion = True
    if not character_motion:
        ground = None
        for obj in scene.objects:
            if obj.get("mo_role") == "ground":
                ground = obj
                break
        if ground:
            scene.frame_set(frame_start)
            ground.scale = ground.scale * 1.0
            ground.keyframe_insert(data_path="scale")
            scene.frame_set(frame_end)
            ground.scale = ground.scale * 1.03
            ground.keyframe_insert(data_path="scale")
            object_motion = True
    for light in [obj for obj in scene.objects if obj.type == "LIGHT"]:
        scene.frame_set(frame_start)
        light.data.energy = light.data.energy * 0.9
        light.data.keyframe_insert(data_path="energy")
        scene.frame_set(frame_end)
        light.data.energy = light.data.energy * 1.1
        light.data.keyframe_insert(data_path="energy")
        light_motion = True
        break
    if not any([camera_motion, character_motion, object_motion, light_motion]):
        raise RuntimeError("No motion sources available for render.")
    scene.frame_set(frame_start)
    return {
        "camera": camera_motion,
        "character": character_motion,
        "object": object_motion,
        "light": light_motion,
    }


def _add_vfx(
    assets_dir: Path,
    scene: bpy.types.Scene,
    camera: bpy.types.Object | None,
    frame_end: int,
    emission_strength: float,
    vfx_scale: float,
    screen_coverage: float,
    warnings: list[str],
) -> None:
    if camera is None or camera.data is None:
        return
    asset_dirs = _resolve_asset_dirs(assets_dir)
    vfx_dir = asset_dirs["vfx"] or (assets_dir / "vfx")
    vfx_collection = _ensure_vfx_collection(scene)
    vfx_items = [
        ("explosion.png", (1.5, 0.0, 1.2)),
        ("energy_arc.png", (0.6, 0.3, 1.0)),
        ("smoke.png", (0.0, -0.2, 0.8)),
    ]
    forward = camera.matrix_world.to_quaternion() @ Vector((0.0, 0.0, -1.0))
    right = camera.matrix_world.to_quaternion() @ Vector((1.0, 0.0, 0.0))
    up = camera.matrix_world.to_quaternion() @ Vector((0.0, 1.0, 0.0))
    distance = 3.0
    camera_pos = camera.location
    vfx_center = camera_pos + forward * distance
    vfx_offset = right * 0.25
    coverage = max(0.1, min(screen_coverage, 0.9))
    vfx_width = 2.0 * distance * math.tan(camera.data.angle / 2.0) * coverage
    vfx_height = vfx_width * (scene.render.resolution_y / scene.render.resolution_x)
    scale_x = vfx_width * 0.5 * vfx_scale
    scale_y = vfx_height * 0.5 * vfx_scale
    for filename, location in vfx_items:
        image_path = vfx_dir / filename
        if not image_path.exists():
            continue
        item_offset = right * (location[0] * 0.15) + up * (location[1] * 0.15)
        bpy.ops.mesh.primitive_plane_add(size=1.0, location=vfx_center + vfx_offset + item_offset)
        plane = bpy.context.active_object
        try:
            _apply_vfx_material(plane, image_path, emission_strength, warnings)
        except Exception as exc:  # noqa: BLE001
            print(f"[VFX] Skipped due to error: {exc}")
            warnings.append("vfx_apply_failed")
            continue
        plane.rotation_euler = camera.rotation_euler
        plane.scale = (scale_x, scale_y, 1.0)
        plane.location = vfx_center + vfx_offset + item_offset
        plane.keyframe_insert(data_path="location", frame=1)
        plane.keyframe_insert(data_path="scale", frame=1)
        plane.keyframe_insert(data_path="location", frame=frame_end)
        plane.keyframe_insert(data_path="scale", frame=frame_end)
        if hasattr(plane, "hide_shadow"):
            plane.hide_shadow = True
        if hasattr(plane, "visible_shadow"):
            plane.visible_shadow = False
        plane.hide_viewport = False
        plane.hide_render = False
        try:
            if plane.name not in vfx_collection.objects:
                vfx_collection.objects.link(plane)
        except Exception:  # noqa: BLE001
            pass
        try:
            if plane.name in scene.collection.objects:
                scene.collection.objects.unlink(plane)
        except Exception:  # noqa: BLE001
            pass


def _setup_compositor(scene: bpy.types.Scene, warnings: list[str]) -> None:
    tree = _get_scene_node_tree(scene)
    print("[POSTFX] tree=", tree)
    if tree is None:
        print("[WARN] Compositor node_tree not available; skipping postfx.")
        warnings.append("postfx_node_tree_missing")
        return
    tree.nodes.clear()
    render_layers = tree.nodes.new(type="CompositorNodeRLayers")
    glare = tree.nodes.new(type="CompositorNodeGlare")
    glare.glare_type = "FOG_GLOW"
    glare.quality = "MEDIUM"
    composite = tree.nodes.new(type="CompositorNodeComposite")
    tree.links.new(render_layers.outputs["Image"], glare.inputs["Image"])
    tree.links.new(glare.outputs["Image"], composite.inputs["Image"])


def _get_scene_node_tree(scene: bpy.types.Scene) -> bpy.types.NodeTree | None:
    try:
        bpy.context.view_layer.update()
    except Exception:  # noqa: BLE001
        pass
    try:
        scene.use_nodes = True
    except Exception:  # noqa: BLE001
        return None
    tree = None
    try:
        tree = bpy.context.scene.node_tree
    except Exception:  # noqa: BLE001
        tree = None
    if tree is None:
        try:
            tree = scene.node_tree
        except Exception:  # noqa: BLE001
            tree = None
    return tree


def _configure_eevee(scene: bpy.types.Scene, quality: str) -> None:
    eevee = getattr(scene, "eevee", None)
    if not eevee:
        return
    major_version = bpy.app.version[0]
    bloom_enabled = False
    if _safe_set(eevee, "use_bloom", True):
        bloom_enabled = True
    elif major_version >= 5:
        print("[WARN] EEVEE bloom not available on this Blender version; continuing.")
    _safe_set(eevee, "bloom_intensity", 0.05)
    _safe_set(eevee, "use_motion_blur", True)
    if quality == "max":
        _safe_set(eevee, "taa_render_samples", 64)
        _safe_set(eevee, "shadow_cube_size", "2048")
        _safe_set(eevee, "shadow_cascade_size", "2048")
    else:
        _safe_set(eevee, "taa_render_samples", 32)
    if major_version >= 5 and not bloom_enabled:
        pass


def _configure_phase15_cycles(scene: bpy.types.Scene, args: argparse.Namespace) -> dict[str, object]:
    scene.render.engine = "CYCLES"
    cycles_prefs = bpy.context.preferences.addons.get("cycles")
    if cycles_prefs is None:
        raise RuntimeError("Cycles addon not available; cannot configure GPU rendering")
    prefs = cycles_prefs.preferences
    try:
        prefs.get_devices()
    except Exception:  # noqa: BLE001
        pass
    backend = None
    for candidate in ("OPTIX", "CUDA"):
        if any(device.type == candidate for device in prefs.devices):
            backend = candidate
            break
    if backend is None:
        raise RuntimeError("No GPU devices available")
    prefs.compute_device_type = backend
    enabled_devices: list[str] = []
    for device in prefs.devices:
        if device.type == "CPU":
            device.use = False
        else:
            device.use = True
        if device.use:
            enabled_devices.append(f"{device.name}:{device.type}")
    if not enabled_devices:
        raise RuntimeError("No GPU devices available")
    scene.cycles.device = "GPU"
    _safe_set(scene.cycles, "samples", args.phase15_samples)
    _safe_set(scene.cycles, "use_adaptive_sampling", True)
    _safe_set(scene.cycles, "adaptive_threshold", 0.01)
    _safe_set(scene.cycles, "max_bounces", args.phase15_bounces)
    _safe_set(scene.cycles, "caustics_reflective", False)
    _safe_set(scene.cycles, "caustics_refractive", False)
    _safe_set(scene.cycles, "use_denoising", True)
    denoise_enabled = False
    if backend == "OPTIX":
        denoise_enabled = _safe_set(scene.cycles, "denoiser", "OPTIX")
    if not denoise_enabled:
        _safe_set(scene.cycles, "denoiser", "OPENIMAGEDENOISE")
    _safe_set(scene.render, "use_persistent_data", True)
    _safe_set(scene.render, "use_motion_blur", False)
    _safe_set(scene.view_settings, "exposure", 0.9)
    _safe_set(scene.view_settings, "gamma", 1.0)
    if hasattr(scene.cycles, "tile_size"):
        _safe_set(scene.cycles, "tile_size", args.phase15_tile)
    print(
        "[ANIME3D_RENDER] "
        f"engine=CYCLES backend={backend} enabled_devices={enabled_devices}"
    )
    return {
        "device_backend": backend.lower(),
        "enabled_devices": enabled_devices,
        "samples": args.phase15_samples,
        "bounces": args.phase15_bounces,
        "denoise": True,
    }


def _write_report(report_path: Path, payload: dict[str, object]) -> None:
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    _phase3_log(
        "PHASE3_BLENDER_RENDER_SEGMENT_ENTER "
        f"segment={Path(args.output).stem} preset={args.render_preset} "
        f"engine={args.engine} device={args.gpu} frame_range=1-{int(max(1, (args.duration or 0) * args.fps) or args.fps)}"
    )
    print(bpy.app.version_string)
    print(f"Render engine: {args.engine}")
    output_path = Path(args.output)
    report_path = Path(args.report)
    assets_dir = Path(args.assets_dir)
    _ensure_parent(output_path)
    _ensure_parent(report_path)
    seed_value = _derive_seed(output_path, args.seed)
    _seed_randomness(seed_value)
    rng = random.Random(seed_value)
    fingerprint = args.fingerprint.strip() or str(seed_value)
    missing_assets = _find_missing_assets(assets_dir)
    strict_assets = int(args.strict_assets) == 1
    procedural_fallback = False
    used_assets: list[str] = []

    assets_inventory, asset_dirs = discover_assets(assets_dir)

    env_candidates = assets_inventory.get("envs", [])
    char_candidates = assets_inventory.get("characters", [])
    anim_candidates = assets_inventory.get("anims", [])
    vfx_candidates = assets_inventory.get("vfx", [])
    env_blend = _select_env_blend(env_candidates, args.environment)
    selected_env = env_blend.stem if env_blend else args.environment
    hero_asset, enemy_asset = _select_character_assets(char_candidates)
    anim_selections = _select_animation_assets(anim_candidates)

    if strict_assets:
        strict_missing = []
        if env_blend is None:
            strict_missing.append("env blend")
        if hero_asset is None:
            strict_missing.append("character blend")
        if anim_selections.get("idle") is None:
            strict_missing.append("idle.fbx")
        if anim_selections.get("run") is None:
            strict_missing.append("run.fbx")
        if anim_selections.get("punch") is None:
            strict_missing.append("punch.fbx")
        if strict_missing:
            error = strict_assets_error(assets_dir, asset_dirs, assets_inventory)
            _write_report(
                report_path,
                {
                    "status": "error",
                    "error": error,
                    "seed": seed_value,
                    "fingerprint": fingerprint,
                    "assets_dir": str(assets_dir),
                    "missing_assets": strict_missing,
                    "used_assets": [],
                    "procedural_fallback": False,
                },
            )
            print(f"[ASSETS] {error}", file=sys.stderr)
            raise SystemExit(2)

    if not strict_assets and (env_blend is None or hero_asset is None or any(v is None for v in anim_selections.values())):
        procedural_fallback = True

    used_assets = [
        str(asset)
        for asset in (env_blend, hero_asset, enemy_asset)
        if asset is not None
    ]

    assets_log = (
        f"[ASSETS] assets_dir={assets_dir} "
        f"env_dir={asset_dirs.get('envs')} char_dir={asset_dirs.get('characters')} "
        f"anim_dir={asset_dirs.get('anims')} vfx_dir={asset_dirs.get('vfx')} "
        f"found_env={len(env_candidates)} found_chars={len(char_candidates)} "
        f"found_anims={len(anim_candidates)} found_vfx={len(vfx_candidates)}"
    )
    char_preview = [path.name for path in char_candidates[:5]]
    assets_log += f" char_preview={char_preview}"
    selected_log = (
        "[ASSETS] selected_env="
        f"{env_blend.name if env_blend else 'none'} "
        f"selected_chars={[p.name for p in (hero_asset, enemy_asset) if p]} "
        f"selected_anims={{"
        f"idle={anim_selections.get('idle').name if anim_selections.get('idle') else 'none'}, "
        f"run={anim_selections.get('run').name if anim_selections.get('run') else 'none'}, "
        f"punch={anim_selections.get('punch').name if anim_selections.get('punch') else 'none'}"
        "}}"
    )
    print(assets_log)
    print(selected_log)
    _phase3_log("PHASE3_ASSET_LOADING_DONE")

    _clear_scene()
    scene = bpy.context.scene
    warnings: list[str] = []
    preset = args.render_preset
    phase15 = preset == "phase15_quality"
    phase15_info: dict[str, object] | None = None
    quality_enabled = os.getenv("MONEYOS_ANIME3D_QUALITY", "1") != "0"
    force_gpu = os.getenv("MONEYOS_ANIME3D_FORCE_GPU", "1") != "0"
    light_preset = os.getenv("MONEYOS_ANIME3D_LIGHT_PRESET", "default").strip().lower()
    outlines_mode = os.getenv("MONEYOS_ANIME3D_OUTLINES", "freestyle")
    compositor_enabled = os.getenv("MONEYOS_ANIME3D_COMPOSITOR", "1") != "0"
    watermark_enabled = os.getenv("MONEYOS_ANIME3D_WATERMARK", "0") != "0"
    try:
        samples = int(os.getenv("MONEYOS_ANIME3D_SAMPLES", "96"))
    except ValueError:
        samples = 96
    tmp_dir = output_path.parent / "tmp" / f"seed_{seed_value}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        bpy.context.preferences.filepaths.temporary_directory = str(tmp_dir)
    except Exception:  # noqa: BLE001
        pass
    try:
        bpy.app.tempdir = str(tmp_dir)
    except Exception:  # noqa: BLE001
        pass

    gpu_info = {
        "requested": False,
        "compute_device_type": "NONE",
        "devices": [],
        "scene_device": "CPU",
    }
    if args.fast_proof:
        engine = "BLENDER_EEVEE_NEXT"
        try:
            scene.render.engine = engine
        except Exception:  # noqa: BLE001
            scene.render.engine = "BLENDER_EEVEE"
        scene.render.fps = 30
    else:
        scene.render.engine = "BLENDER_EEVEE" if args.engine == "eevee" else "CYCLES"
        scene.render.fps = args.fps
    if (args.engine or "").lower() == "cycles" and str(args.gpu) == "1":
        gpu_info = _configure_cycles_gpu_optix(scene, args)
    if args.duration is None or args.duration <= 0:
        raise RuntimeError("Duration must be provided and > 0.")
    total_frames = max(1, int(round(args.duration * args.fps)))
    scene.frame_start = 1
    scene.frame_end = total_frames
    scene.frame_set(0)
    _apply_resolution(scene, args)
    scene.render.image_settings.file_format = "PNG"
    scene.render.use_file_extension = True
    frames_dir = output_path.parent / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(frames_dir / "frame_######")

    scene.render.use_freestyle = args.outline_mode == "freestyle"
    if hasattr(scene.render, "line_thickness"):
        scene.render.line_thickness = 1.5
    if phase15:
        phase15_info = _configure_phase15_cycles(scene, args)
        print(
            "[PHASE15] engine=cycles "
            f"device={phase15_info['device_backend']} "
            f"samples={phase15_info['samples']} "
            f"res={scene.render.resolution_x}x{scene.render.resolution_y} "
            f"fps={scene.render.fps} duration={args.duration:.2f}"
        )
    elif hasattr(scene, "eevee"):
        _configure_eevee(scene, args.quality)

    procedural_humanoid = False
    if procedural_fallback:
        if strict_assets:
            raise RuntimeError("Missing assets; Blender-only pipeline requires asset packs.")
        raise RuntimeError(
            "Missing assets even after auto-install attempt. "
            "Check server logs and MONEYOS_ASSET_PACK_URLS."
        )

    beat_plan = []
    if args.beat_plan:
        plan_path = Path(args.beat_plan)
        if plan_path.exists():
            payload = json.loads(plan_path.read_text(encoding="utf-8"))
            beat_plan = payload.get("plan", []) if isinstance(payload, dict) else []
    shot_plan = _build_shot_plan(beat_plan, args.fps, total_frames, seed_value)
    shot_plan_path = output_path.parent / "shot_plan.json"
    shot_plan_path.write_text(json.dumps({"shots": shot_plan}, indent=2), encoding="utf-8")
    cut_points = [int(shot["end_frame"]) for shot in shot_plan[:-1]]
    print(f"[DIRECTOR] shots={len(shot_plan)} cuts={cut_points}")
    if args.asset_mode != "local":
        template_options = ["room", "street", "studio"]
        if args.environment and args.environment not in template_options:
            template_options.append(args.environment)
        selected_env = rng.choice(template_options)
        if beat_plan:
            selected_env = _apply_environment_schedule(beat_plan, args.fps, selected_env)
        else:
            _build_environment_template(selected_env)
    else:
        if beat_plan and env_candidates:
            selected_env = _apply_environment_schedule_local(beat_plan, args.fps, env_candidates, selected_env)
    print(f"[PHASE2] env={selected_env} character={args.character_asset or 'none'} preset={preset}")
    objects = _create_scene(
        assets_dir,
        args.asset_mode,
        env_blend,
        hero_asset,
        enemy_asset,
        selected_env,
        style_preset=args.style_preset,
    )
    _phase3_log("PHASE3_CHARACTER_IMPORT_DONE")
    character_meshes = _load_character_asset(assets_dir, args.character_asset, warnings)
    if character_meshes:
        for obj in character_meshes:
            obj["mo_role"] = "subject"
    style_counts = {"characters": 0, "props": 0, "meshes_styled": 0}
    if str(args.style_preset).strip().lower() == "anime_visual":
        _phase3_log("PHASE3_SHADER_APPLY_START")
        style_counts = _apply_anime_visual_style(scene, assets_dir, args.outline_mode)
        _phase3_log("PHASE3_SHADER_APPLY_DONE")
    material_issues = _phase3_validate_materials(scene)
    if material_issues:
        _phase3_log(
            "PHASE3_SHADER_MISSING_OR_INVALID "
            f"count={len(material_issues)} sample={material_issues[:5]}"
        )
    _ensure_visual_density(scene, args.duration, args.fps)
    if args.character_variation:
        try:
            variation = json.loads(args.character_variation)
        except json.JSONDecodeError:
            variation = {}
        _apply_character_variation(scene, variation)
    if str(args.style_preset).strip().lower() == "anime_visual":
        try:
            from app.core.validators.anime_character_validator import validate_anime_character_scene

            validate_anime_character_scene(scene)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Anime character validation failed: {exc}") from exc
    phase3_character_check = _phase3_character_style_check(scene)
    visibility_info = _setup_visibility_scene(scene, objects.get("camera"), rng)
    _phase3_log("PHASE3_CAMERA_SETUP_DONE")
    _add_vfx(
        assets_dir,
        scene,
        objects.get("camera"),
        scene.frame_end,
        args.vfx_emission_strength,
        args.vfx_scale,
        args.vfx_screen_coverage,
        warnings,
    )
    print("[POSTFX] enabled=", args.postfx)
    if args.postfx == "on" and not args.fast_proof:
        _setup_compositor(scene, warnings)
    envelope = _load_rms_envelope(Path(args.audio) if args.audio else Path(), args.fps, scene.frame_end)
    dialogue_text = " ".join(str(beat.get("dialogue", "")) for beat in beat_plan) if beat_plan else ""
    viseme_schedule = _build_viseme_schedule(dialogue_text, scene.frame_end)
    motion_info = _ensure_minimum_motion(objects, scene, args.duration, args.fps)
    print(
        "[MOTION] camera_motion="
        f"{motion_info['camera']} character_motion={motion_info['character']} "
        f"object_motion={motion_info['object']} light_motion={motion_info['light']}"
    )
    fast_proof_like = args.fast_proof or phase15
    mouth_keyframes = _animate(
        objects,
        envelope,
        viseme_schedule,
        args.fps,
        assets_dir,
        args.asset_mode,
        fast_proof_like,
        args.mode,
    )
    if quality_enabled:
        subject_obj, char_source, char_fmt = _ensure_character(scene, args, assets_dir, seed_value)
        _setup_anime_lighting(scene, subject_obj)
        setup_anime_materials(scene.collection, preset=light_preset)
        if compositor_enabled:
            _setup_anime_compositor(scene, args)
        camera_modes = ["push_in", "orbit", "handheld", "static"]
        camera_mode = camera_modes[seed_value % len(camera_modes)]
        camera_obj = objects.get("camera") if isinstance(objects, dict) else scene.camera
        if camera_obj:
            setup_anime_camera_motion(camera_obj, subject_obj, camera_mode, total_frames, seed_value)
        rig_obj = objects.get("rig") if isinstance(objects, dict) else None
        apply_anime_animation_polish(rig_obj, total_frames, seed_value)
        _apply_outlines(scene, outlines_mode)
        if watermark_enabled:
            _apply_watermark(
                scene,
                f"Anime3D | 1080p | S{samples} | CHAR:{char_fmt} | SRC:{char_source}",
            )

    subject_obj = _get_subject_object(scene)
    camera_obj = objects.get("camera") if isinstance(objects, dict) else scene.camera
    if camera_obj is None:
        camera_obj = scene.camera

    selection = {
        "seed": seed_value,
        "assets_dir": str(assets_dir),
        "selected_environment": selected_env,
        "selected_environment_blend": str(env_blend) if env_blend else None,
        "selected_characters": [
            str(hero_asset) if hero_asset else None,
            str(enemy_asset) if enemy_asset else None,
        ],
        "camera_preset": visibility_info.get("camera_preset"),
        "camera_params": visibility_info.get("camera_params"),
        "key_light_params": visibility_info.get("key_light_params"),
        "mode": args.mode,
        "style_preset": args.style_preset,
        "anime_visual_assets": style_counts,
    }
    chosen_variation = {
        "camera_variant": visibility_info.get("camera_variant"),
        "light_variant": visibility_info.get("light_variant"),
        "hue_variant": visibility_info.get("hue_variant"),
    }
    print(
        "[FINGERPRINT] "
        f"seed={seed_value} fp={fingerprint} env={selected_env} "
        f"mode={args.mode} assets={assets_dir}"
    )
    _write_report(
        report_path,
        {
            "status": "started",
            "fingerprint": fingerprint,
            "parsed_args": vars(args),
            "chosen_variation": chosen_variation,
            "missing_assets": missing_assets,
            "used_assets": used_assets,
            "procedural_fallback": False,
            "shot_plan": str(shot_plan_path),
            **selection,
        },
    )

    for idx, shot in enumerate(shot_plan, start=1):
        shot_start = int(shot["start_frame"])
        shot_end = int(shot["end_frame"])
        scene.frame_start = shot_start
        scene.frame_end = shot_end
        if camera_obj:
            _apply_shot_camera(scene, camera_obj, subject_obj, shot, next_shot=shot_plan[idx] if idx < len(shot_plan) else None)
        lighting_preset = str(shot.get("lighting_preset", "DARK_CONTRAST"))
        lighting_intensity = float(shot.get("lighting_intensity", 1.0))
        lighting_color = str(shot.get("lighting_color", "#FFFFFF"))
        _apply_lighting_preset(scene, f"{lighting_preset}|intensity={lighting_intensity}|color={lighting_color}")
        print(f"[LIGHTING] preset = {lighting_preset} shot={idx}/{len(shot_plan)}")
        if idx == 1:
            _phase3_log("PHASE3_LIGHTING_SETUP_DONE")
        _apply_impact_vfx(scene, shot, args.vfx_emission_strength)
        print(
            f"[SHOT {idx}/{len(shot_plan)}] pre-hold={shot.get('pre_hold_frames', 0)} "
            f"post-ease={shot.get('post_ease_frames', 0)} preset={shot.get('camera_preset')} "
            f"frames={shot_start}-{shot_end} impact_hold={shot.get('hold_frames', 0)}"
        )
        bpy.ops.render.render(animation=True, write_still=False)

    scene.frame_start = 1
    scene.frame_end = total_frames
    rendered_report = {
        "status": "rendered",
        "frame_count": scene.frame_end,
        "frames_dir": str(frames_dir),
        "seed": seed_value,
        "fingerprint": fingerprint,
        "shot_plan": str(shot_plan_path),
    }
    report_path.write_text(json.dumps(rendered_report, indent=2), encoding="utf-8")

    render_report = {
        "status": "complete",
        "fingerprint": fingerprint,
        "mouth_keyframes": mouth_keyframes,
        "frame_end": scene.frame_end,
        "fps": scene.render.fps,
        "vfx_emission_strength": args.vfx_emission_strength,
        "vfx_scale": args.vfx_scale,
        "vfx_screen_coverage": args.vfx_screen_coverage,
        "warnings": warnings,
        "parsed_args": vars(args),
        "preset": preset,
        "engine": scene.render.engine,
        "device": phase15_info["device_backend"] if phase15_info else None,
        "samples": phase15_info["samples"] if phase15_info else None,
        "bounces": phase15_info["bounces"] if phase15_info else None,
        "denoise": phase15_info["denoise"] if phase15_info else None,
        "res": f"{scene.render.resolution_x}x{scene.render.resolution_y}",
        "duration": args.duration,
        "environment": selected_env,
        "character_asset": args.character_asset or None,
        "subject_bbox": visibility_info.get("subject_bbox"),
        "camera_params": visibility_info.get("camera_params"),
        "world_strength": visibility_info.get("world_strength"),
        "chosen_variation": chosen_variation,
        "seed": seed_value,
        "assets_dir": str(assets_dir),
        "selected_environment": selected_env,
        "selected_environment_blend": str(env_blend) if env_blend else None,
        "selected_characters": selection["selected_characters"],
        "camera_preset_name": visibility_info.get("camera_preset"),
        "key_light_params": visibility_info.get("key_light_params"),
        "mode_used": args.mode,
        "mode": args.mode,
        "style_preset": args.style_preset,
        "anime_visual_assets": style_counts,
        "gpu": gpu_info,
        "shot_plan": str(shot_plan_path),
        "shot_count": len(shot_plan),
        "procedural_humanoid": procedural_humanoid,
        "missing_assets": missing_assets,
        "used_assets": used_assets,
        "procedural_fallback": False,
        "debug": {
            "phase3_character_check": phase3_character_check,
            "phase3_material_issues": material_issues,
        },
    }
    report_path.write_text(json.dumps(render_report, indent=2), encoding="utf-8")
    _phase3_log(
        "PHASE3_BLENDER_RENDER_SEGMENT_EXIT "
        f"success=1 output={output_path} frames={scene.frame_end}"
    )


if __name__ == "__main__":
    main()
