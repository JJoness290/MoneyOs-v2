from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import wave
from pathlib import Path

import bpy
from mathutils import Vector


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
    parser.add_argument("--character-asset", default="")
    parser.add_argument("--mode", default="default")
    parser.add_argument("--style-preset", default="key_art")
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
                principled.inputs["Specular"].default_value = 0.25
                principled.inputs["Emission Strength"].default_value = 0.03
            name_lower = mat.name.lower()
            if any(tag in name_lower for tag in ("eye", "iris", "pupil")) and principled:
                principled.inputs["Roughness"].default_value = 0.2
                principled.inputs["Specular"].default_value = 0.5
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
    for fcurve in camera.animation_data.action.fcurves:
        for keyframe in fcurve.keyframe_points:
            keyframe.interpolation = "BEZIER"
            keyframe.handle_left_type = "AUTO_CLAMPED"
            keyframe.handle_right_type = "AUTO_CLAMPED"
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
            for fcurve in obj.animation_data.action.fcurves:
                for keyframe in fcurve.keyframe_points:
                    keyframe.interpolation = "BEZIER"
                    keyframe.handle_left_type = "AUTO_CLAMPED"
                    keyframe.handle_right_type = "AUTO_CLAMPED"
    print("[ANIM] polish applied")


def _setup_anime_compositor(scene: bpy.types.Scene, args: argparse.Namespace) -> None:
    if args.postfx != "on":
        return
    scene.use_nodes = True
    tree = scene.node_tree
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


def _discover_assets(assets_dir: Path) -> dict[str, list[Path]]:
    envs = sorted((assets_dir / "envs").glob("*.blend"))
    characters = sorted((assets_dir / "characters").glob("*.blend"))
    anims = sorted((assets_dir / "anims").glob("*.fbx"))
    vfx = sorted((assets_dir / "vfx").glob("*.*"))
    return {
        "envs": envs,
        "characters": characters,
        "anims": anims,
        "vfx": vfx,
    }


def _required_assets(assets_dir: Path) -> dict[str, Path]:
    return {
        "characters/hero.blend": assets_dir / "characters" / "hero.blend",
        "characters/enemy.blend": assets_dir / "characters" / "enemy.blend",
        "envs/city.blend": assets_dir / "envs" / "city.blend",
        "anims/idle.fbx": assets_dir / "anims" / "idle.fbx",
        "anims/run.fbx": assets_dir / "anims" / "run.fbx",
        "anims/punch.fbx": assets_dir / "anims" / "punch.fbx",
        "vfx/explosion.png": assets_dir / "vfx" / "explosion.png",
        "vfx/energy_arc.png": assets_dir / "vfx" / "energy_arc.png",
        "vfx/smoke.png": assets_dir / "vfx" / "smoke.png",
    }


def _find_missing_assets(assets_dir: Path) -> list[str]:
    return [key for key, path in _required_assets(assets_dir).items() if not path.exists()]


def _create_procedural_humanoid(name: str, location: tuple[float, float, float]) -> tuple[bpy.types.Object, int]:
    print(f"[ANIME3D_CHAR] source=generated name={name}")
    bpy.ops.object.empty_add(type="PLAIN_AXES", location=location)
    root = bpy.context.active_object
    root.name = name
    material = bpy.data.materials.new(name=f"{name}_ProceduralMaterial")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    principled = nodes.get("Principled BSDF")
    if principled:
        principled.inputs["Base Color"].default_value = (0.78, 0.7, 0.66, 1.0)
        principled.inputs["Roughness"].default_value = 0.25
        principled.inputs["Emission"].default_value = (0.08, 0.06, 0.05, 1.0)
        principled.inputs["Emission Strength"].default_value = 0.08
    bpy.ops.mesh.primitive_cylinder_add(radius=0.35, depth=1.1, location=(location[0], location[1], 0.95))
    torso = bpy.context.active_object
    torso.name = f"{name}_Torso"
    torso.scale.x = 0.85
    torso.scale.y = 0.6
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.25, location=(location[0], location[1], 1.7))
    head = bpy.context.active_object
    head.name = f"{name}_Head"
    bpy.ops.mesh.primitive_cylinder_add(radius=0.08, depth=0.7, location=(location[0] + 0.45, location[1], 1.2))
    arm_r = bpy.context.active_object
    arm_r.name = f"{name}_Arm_R"
    arm_r.rotation_euler = (0.0, math.radians(90), 0.0)
    bpy.ops.mesh.primitive_cylinder_add(radius=0.08, depth=0.7, location=(location[0] - 0.45, location[1], 1.2))
    arm_l = bpy.context.active_object
    arm_l.name = f"{name}_Arm_L"
    arm_l.rotation_euler = (0.0, math.radians(90), 0.0)
    bpy.ops.mesh.primitive_cylinder_add(radius=0.1, depth=0.8, location=(location[0] + 0.18, location[1], 0.35))
    leg_r = bpy.context.active_object
    leg_r.name = f"{name}_Leg_R"
    bpy.ops.mesh.primitive_cylinder_add(radius=0.1, depth=0.8, location=(location[0] - 0.18, location[1], 0.35))
    leg_l = bpy.context.active_object
    leg_l.name = f"{name}_Leg_L"
    for obj in (torso, head, arm_r, arm_l, leg_r, leg_l):
        obj.parent = root
        if obj.data.materials:
            obj.data.materials[0] = material
        else:
            obj.data.materials.append(material)
    if name.lower() == "hero":
        for obj in (torso, head, arm_r, arm_l, leg_r, leg_l):
            obj["mo_role"] = "subject"
    return root, 6


def _animate_procedural_humanoid(root: bpy.types.Object, total_frames: int) -> None:
    if total_frames <= 1:
        return
    start = 1
    mid = max(2, total_frames // 2)
    end = total_frames
    root.location.y = root.location.y
    root.keyframe_insert(data_path="location", frame=start)
    root.location.y += 0.05
    root.keyframe_insert(data_path="location", frame=mid)
    root.location.y -= 0.05
    root.keyframe_insert(data_path="location", frame=end)
    for child in root.children:
        if "Arm_" not in child.name:
            continue
        child.rotation_euler.z = 0.0
        child.keyframe_insert(data_path="rotation_euler", frame=start)
        child.rotation_euler.z = math.radians(4)
        child.keyframe_insert(data_path="rotation_euler", frame=mid)
        child.rotation_euler.z = math.radians(-4)
        child.keyframe_insert(data_path="rotation_euler", frame=end)


def _build_procedural_scene(
    scene: bpy.types.Scene,
    total_frames: int,
    force_procedural_humanoid: bool,
) -> bool:
    bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, 0))
    floor = bpy.context.active_object
    floor.name = "Floor"
    bpy.ops.object.light_add(type="AREA", location=(0, -3, 4))
    light = bpy.context.active_object
    light.data.energy = 600
    bpy.ops.object.camera_add(location=(0, -6, 2), rotation=(1.3, 0, 0))
    camera = bpy.context.active_object
    scene.camera = camera
    scene.frame_start = 1
    scene.frame_end = total_frames
    procedural_humanoid = False
    if force_procedural_humanoid:
        procedural_humanoid = True
        hero, hero_parts = _create_procedural_humanoid("Hero", (-1.5, 0, 0))
        enemy, enemy_parts = _create_procedural_humanoid("Enemy", (1.5, 0, 0))
        print(
            "[ANIME3D_CHAR] "
            f"procedural_humanoid=1 created=Hero parts={hero_parts} "
            f"created=Enemy parts={enemy_parts}"
        )
    else:
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(-1.5, 0, 0.5))
        hero = bpy.context.active_object
        hero.name = "Hero"
        hero["mo_role"] = "subject"
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(1.5, 0, 0.5))
        enemy = bpy.context.active_object
        enemy.name = "Enemy"
    _animate_procedural_humanoid(hero, total_frames)
    _animate_procedural_humanoid(enemy, total_frames)
    hero.location = (-1.5, 0, 0)
    hero.keyframe_insert(data_path="location", frame=1)
    hero.location = (-0.4, 0.0, 0)
    hero.keyframe_insert(data_path="location", frame=total_frames)
    enemy.location = (1.5, 0, 0)
    enemy.keyframe_insert(data_path="location", frame=1)
    enemy.location = (0.4, 0.0, 0)
    enemy.keyframe_insert(data_path="location", frame=total_frames)
    camera.location = (0, -6, 2)
    camera.keyframe_insert(data_path="location", frame=1)
    camera.location = (0, -5, 2.5)
    camera.keyframe_insert(data_path="location", frame=total_frames)
    return procedural_humanoid


def _find_character_asset(workdir: Path) -> Path | None:
    search_dirs = [workdir, workdir / "assets", workdir / "characters"]
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
        return None, "vrm_unsupported"
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
        for ext in (".blend", ".fbx", ".glb", ".gltf"):
            matches = sorted(chars_dir.glob(f"*{ext}"))
            if matches:
                asset_path = matches[0]
                break
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
    _create_procedural_humanoid("Hero", (0.0, 0.0, 0.0))
    subject = _get_subject_object(scene)
    if subject is None:
        raise RuntimeError("Unable to generate procedural character")
    return subject, "generated", "procedural"


def _apply_outlines(scene: bpy.types.Scene, mode: str) -> None:
    mode = mode.strip().lower()
    if mode == "off":
        scene.render.use_freestyle = False
        return
    if mode == "freestyle":
        scene.render.use_freestyle = True
        if hasattr(scene.render, "line_thickness"):
            scene.render.line_thickness = 1.5
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


def _apply_toon_material(obj: bpy.types.Object, outline_material: bpy.types.Material) -> None:
    if obj.type != "MESH":
        return
    material = bpy.data.materials.new(name="ToonMaterial")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new(type="ShaderNodeOutputMaterial")
    toon = nodes.new(type="ShaderNodeBsdfToon")
    toon.inputs["Size"].default_value = 0.7
    toon.inputs["Smooth"].default_value = 0.05
    rim = nodes.new(type="ShaderNodeFresnel")
    rim.inputs["IOR"].default_value = 1.3
    mix = nodes.new(type="ShaderNodeMixShader")
    emission = nodes.new(type="ShaderNodeEmission")
    emission.inputs["Color"].default_value = (0.4, 0.6, 1.0, 1.0)
    emission.inputs["Strength"].default_value = 0.6
    material.node_tree.links.new(toon.outputs["BSDF"], mix.inputs[1])
    material.node_tree.links.new(emission.outputs["Emission"], mix.inputs[2])
    material.node_tree.links.new(rim.outputs["Fac"], mix.inputs[0])
    material.node_tree.links.new(mix.outputs["Shader"], output.inputs["Surface"])
    if obj.data.materials:
        obj.data.materials[0] = material
    else:
        obj.data.materials.append(material)
    _ensure_material(obj.data.materials, outline_material)
    print("[MAT] applied outline material to", obj.name)
    modifier = obj.modifiers.new(name="Outline", type="SOLIDIFY")
    modifier.thickness = 0.02
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
    bpy.ops.mesh.primitive_cylinder_add(radius=0.4, depth=1.2, location=(location[0], location[1], location[2] + 1.2))
    body = bpy.context.active_object
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.45, location=(location[0], location[1], location[2] + 2.2))
    head = bpy.context.active_object
    bpy.ops.mesh.primitive_cube_add(size=0.25, location=(location[0], location[1] + 0.35, location[2] + 2.0))
    jaw = bpy.context.active_object
    if name == "hero":
        body["mo_role"] = "subject"
        head["mo_role"] = "subject"
        jaw["mo_role"] = "subject"
    for offset in (-0.5, 0.5):
        bpy.ops.mesh.primitive_cylinder_add(radius=0.15, depth=0.8, location=(location[0] + offset, location[1], location[2] + 0.6))
        leg = bpy.context.active_object
        leg.rotation_euler.x = math.radians(90)
    for offset in (-0.6, 0.6):
        bpy.ops.mesh.primitive_cylinder_add(radius=0.12, depth=0.9, location=(location[0] + offset, location[1], location[2] + 1.6))
        arm = bpy.context.active_object
        arm.rotation_euler.y = math.radians(90)
    return {"body": body, "head": head, "jaw": jaw}


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
    for fcurve in action.fcurves:
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


def _create_scene(
    assets_dir: Path,
    asset_mode: str,
    env_blend: Path | None,
    hero_asset: Path | None,
    enemy_asset: Path | None,
) -> dict[str, bpy.types.Object | None]:
    scene = bpy.context.scene
    outline_material = _create_outline_material()

    hero_armature = None
    enemy_armature = None
    hero_jaw = None
    hero_body = None
    if asset_mode == "local":
        env_path = env_blend or (assets_dir / "envs" / "city.blend")
        hero_path = hero_asset or (assets_dir / "characters" / "hero.blend")
        enemy_path = enemy_asset or (assets_dir / "characters" / "enemy.blend")
        env_collections = _append_collections(env_path)
        hero_collections = _append_collections(hero_path)
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
        if enemy_armature:
            enemy_armature.location = (3, -2, 0)
    else:
        hero = _create_character("hero", (0, 0, 0))
        enemy = _create_character("enemy", (2.5, -2.0, 0))
        hero_jaw = hero["jaw"]
        hero_body = hero["body"]
        for obj in (hero["body"], hero["head"], hero["jaw"], enemy["body"], enemy["head"], enemy["jaw"]):
            _apply_toon_material(obj, outline_material)

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
        asset_path = assets_dir / "characters" / character_asset
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
        raise RuntimeError("VRM import not installed; use .blend character assets instead.")
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


def _animate(
    objects: dict[str, bpy.types.Object | None],
    envelope: list[float],
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
        action_path = assets_dir / "anims" / "run.fbx"
        if action_path.exists():
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
        if jaw_bone:
            jaw_bone.rotation_euler.x = -envelope[frame] * 0.6
            jaw_bone.keyframe_insert(data_path="rotation_euler", index=0)
            if envelope[frame] > 0.02:
                mouth_keyframes += 1
        elif objects.get("hero_jaw"):
            jaw_obj = objects["hero_jaw"]
            jaw_obj.rotation_euler.x = -envelope[frame] * 0.6
            jaw_obj.keyframe_insert(data_path="rotation_euler", index=0)
            if envelope[frame] > 0.02:
                mouth_keyframes += 1
        elif mouth_key:
            _, key = mouth_key
            key.value = min(1.0, envelope[frame] * 1.2)
            key.keyframe_insert(data_path="value")
            if envelope[frame] > 0.02:
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
        image_path = assets_dir / "vfx" / filename
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

    if missing_assets and strict_assets:
        error = f"missing assets: {', '.join(missing_assets)}"
        _write_report(
            report_path,
            {
                "status": "error",
                "error": error,
                "seed": seed_value,
                "fingerprint": fingerprint,
                "assets_dir": str(assets_dir),
                "missing_assets": missing_assets,
                "used_assets": used_assets,
                "procedural_fallback": False,
            },
        )
        print(f"[ASSETS] {error}", file=sys.stderr)
        raise SystemExit(2)
    if missing_assets and not strict_assets:
        procedural_fallback = True
    assets_inventory = _discover_assets(assets_dir) if args.asset_mode == "local" else {}

    env_candidates = assets_inventory.get("envs", [])
    char_candidates = assets_inventory.get("characters", [])
    selected_env = rng.choice(env_candidates).stem if env_candidates else args.environment
    env_blend = rng.choice(env_candidates) if env_candidates else None
    hero_asset = rng.choice(char_candidates) if char_candidates else None
    enemy_asset = rng.choice(char_candidates) if char_candidates else None
    if hero_asset and enemy_asset and hero_asset == enemy_asset and len(char_candidates) > 1:
        enemy_asset = rng.choice([path for path in char_candidates if path != hero_asset])
    used_assets = [
        str(asset)
        for asset in (env_blend, hero_asset, enemy_asset)
        if asset is not None
    ]

    assets_log = (
        f"[ASSETS] assets_dir={assets_dir} "
        f"found_env={len(env_candidates)} found_chars={len(char_candidates)} "
        f"found_anims={len(assets_inventory.get('anims', []))} "
        f"found_vfx={len(assets_inventory.get('vfx', []))}"
    )
    print(assets_log)

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
    watermark_enabled = os.getenv("MONEYOS_ANIME3D_WATERMARK", "1") != "0"
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
    scene.render.filepath = str(frames_dir / "frame_####")

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
        print("[ASSETS] missing assets detected. Using procedural fallback.")
        procedural_humanoid = _build_procedural_scene(
            scene,
            total_frames,
            force_procedural_humanoid=bool(args.force_procedural_humanoid),
        )
        subject_obj, char_source, char_fmt = _ensure_character(scene, args, assets_dir, seed_value)
        _apply_outlines(scene, outlines_mode)
        if quality_enabled:
            _setup_anime_lighting(scene, subject_obj)
            setup_anime_materials(scene.collection, preset=light_preset)
            if compositor_enabled:
                _setup_anime_compositor(scene, args)
            if scene.camera:
                camera_modes = ["push_in", "orbit", "handheld", "static"]
                camera_mode = camera_modes[seed_value % len(camera_modes)]
                setup_anime_camera_motion(scene.camera, subject_obj, camera_mode, total_frames, seed_value)
        if watermark_enabled:
            _apply_watermark(
                scene,
                f"key_art_v1 | 1080p | S{samples} | CHAR:{char_fmt} | SRC:{char_source}",
            )
        _write_report(
            report_path,
            {
                "status": "started",
                "fingerprint": fingerprint,
                "seed": seed_value,
                "missing_assets": missing_assets,
                "used_assets": used_assets,
                "procedural_fallback": True,
                "parsed_args": vars(args),
            },
        )
        bpy.ops.render.render(animation=True, write_still=False)
        rendered_report = {
            "status": "rendered",
            "frame_count": scene.frame_end,
            "frames_dir": str(frames_dir),
            "seed": seed_value,
            "fingerprint": fingerprint,
        }
        report_path.write_text(json.dumps(rendered_report, indent=2), encoding="utf-8")
        render_report = {
            "status": "complete",
            "fingerprint": fingerprint,
            "seed": seed_value,
            "frame_end": scene.frame_end,
            "fps": scene.render.fps,
            "res": f"{scene.render.resolution_x}x{scene.render.resolution_y}",
            "duration": args.duration,
            "warnings": warnings,
            "parsed_args": vars(args),
            "gpu": gpu_info,
            "procedural_humanoid": procedural_humanoid,
            "missing_assets": missing_assets,
            "used_assets": used_assets,
            "procedural_fallback": True,
        }
        report_path.write_text(json.dumps(render_report, indent=2), encoding="utf-8")
        return

    if args.asset_mode != "local":
        template_options = ["room", "street", "studio"]
        if args.environment and args.environment not in template_options:
            template_options.append(args.environment)
        selected_env = rng.choice(template_options)
    print(f"[PHASE2] env={selected_env} character={args.character_asset or 'none'} preset={preset}")
    if args.asset_mode != "local":
        _build_environment_template(selected_env)
    objects = _create_scene(assets_dir, args.asset_mode, env_blend, hero_asset, enemy_asset)
    _ensure_visual_density(scene, args.duration, args.fps)
    character_meshes = _load_character_asset(assets_dir, args.character_asset, warnings)
    if character_meshes:
        for obj in character_meshes:
            obj["mo_role"] = "subject"
    visibility_info = _setup_visibility_scene(scene, objects.get("camera"), rng)
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
                f"key_art_v1 | 1080p | S{samples} | CHAR:{char_fmt} | SRC:{char_source}",
            )

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
            **selection,
        },
    )

    bpy.ops.render.render(animation=True, write_still=False)
    rendered_report = {
        "status": "rendered",
        "frame_count": scene.frame_end,
        "frames_dir": str(frames_dir),
        "seed": seed_value,
        "fingerprint": fingerprint,
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
        "gpu": gpu_info,
        "procedural_humanoid": procedural_humanoid,
        "missing_assets": missing_assets,
        "used_assets": used_assets,
        "procedural_fallback": False,
    }
    report_path.write_text(json.dumps(render_report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
