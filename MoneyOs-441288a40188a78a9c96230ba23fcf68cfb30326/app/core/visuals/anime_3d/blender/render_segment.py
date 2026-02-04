from __future__ import annotations

import argparse
import json
import math
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
    parser.add_argument("--engine", default="eevee")
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--audio", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--assets-dir", required=True)
    parser.add_argument("--asset-mode", default="auto")
    parser.add_argument("--style-preset", default="key_art")
    parser.add_argument("--outline-mode", default="freestyle")
    parser.add_argument("--postfx", default="on")
    parser.add_argument("--quality", default="balanced")
    parser.add_argument("--res", default="1920x1080")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--vfx-emission-strength", type=float, default=50.0)
    parser.add_argument("--vfx-scale", type=float, default=1.0)
    parser.add_argument("--vfx-screen-coverage", type=float, default=0.35)
    return parser.parse_args(argv)


def _clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in bpy.data.meshes:
        bpy.data.meshes.remove(block)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


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


def _create_scene(assets_dir: Path, asset_mode: str) -> dict[str, bpy.types.Object | None]:
    scene = bpy.context.scene
    outline_material = _create_outline_material()

    hero_armature = None
    enemy_armature = None
    hero_jaw = None
    hero_body = None
    if asset_mode == "local":
        env_collections = _append_collections(assets_dir / "envs" / "city.blend")
        hero_collections = _append_collections(assets_dir / "characters" / "hero.blend")
        enemy_collections = _append_collections(assets_dir / "characters" / "enemy.blend")

        hero_armature = _find_armature(hero_collections)
        enemy_armature = _find_armature(enemy_collections)

        for collection in hero_collections + enemy_collections:
            for obj in collection.all_objects:
                _apply_toon_material(obj, outline_material)

        if hero_armature:
            hero_armature.location = (0, 0, 0)
        if enemy_armature:
            enemy_armature.location = (3, -2, 0)
    else:
        _create_city_env(outline_material)
        hero = _create_character("hero", (0, 0, 0))
        enemy = _create_character("enemy", (2.5, -2.0, 0))
        hero_jaw = hero["jaw"]
        hero_body = hero["body"]
        for obj in (hero["body"], hero["head"], hero["jaw"], enemy["body"], enemy["head"], enemy["jaw"]):
            _apply_toon_material(obj, outline_material)

    bpy.ops.object.light_add(type="SUN", location=(5, -5, 8))
    key_light = bpy.context.active_object
    key_light.data.energy = 4.5
    bpy.ops.object.light_add(type="POINT", location=(-3, 2, 5))
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


def _animate(
    objects: dict[str, bpy.types.Object | None],
    envelope: list[float],
    fps: int,
    assets_dir: Path,
    asset_mode: str,
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

    if camera:
        camera.data.lens = 18
    for frame in range(frame_count):
        bpy.context.scene.frame_set(frame + 1)
        t = frame / fps
        if hero_armature:
            hero_armature.location.x = t * 0.03
            hero_armature.keyframe_insert(data_path="location", index=0)
        elif objects.get("hero_body"):
            hero_body = objects["hero_body"]
            hero_body.rotation_euler.z = math.sin(t * 2.0) * 0.2
            hero_body.keyframe_insert(data_path="rotation_euler", index=2)
        if camera:
            camera.location.x = 4 + math.sin(t * 0.8) * 0.4
            camera.location.y = -6 + math.cos(t * 0.7) * 0.4
            camera.location.z = 2.5 + math.sin(t * 1.2) * 0.2
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
    material.shadow_method = "NONE"
    material.use_backface_culling = False
    if hasattr(material, "alpha_threshold"):
        material.alpha_threshold = 0.0
    if plane.data.materials:
        plane.data.materials[0] = material
    else:
        plane.data.materials.append(material)
    print("[VFX] emissive material applied:", image_path.name, "strength", emission_strength)


def _add_vfx(
    assets_dir: Path,
    scene: bpy.types.Scene,
    camera: bpy.types.Object | None,
    frame_end: int,
    emission_strength: float,
    vfx_scale: float,
    screen_coverage: float,
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
            _apply_vfx_material(plane, image_path, emission_strength)
        except Exception as exc:  # noqa: BLE001
            print(f"[VFX] Skipped due to error: {exc}")
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


def _setup_compositor(scene: bpy.types.Scene) -> None:
    tree = _get_scene_node_tree(scene)
    print("[POSTFX] tree=", tree)
    if tree is None:
        print("[WARN] Compositor node_tree not available; skipping postfx.")
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


def main() -> None:
    args = _parse_args()
    print(bpy.app.version_string)
    print(f"Render engine: {args.engine}")
    output_path = Path(args.output)
    report_path = Path(args.report)
    assets_dir = Path(args.assets_dir)
    _ensure_parent(output_path)
    _ensure_parent(report_path)
    report_path.write_text(
        json.dumps({"parsed_args": vars(args)}, indent=2),
        encoding="utf-8",
    )
    if args.asset_mode == "local":
        required = [
            assets_dir / "characters" / "hero.blend",
            assets_dir / "envs" / "city.blend",
            assets_dir / "anims" / "run.fbx",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise RuntimeError(f"Missing assets (assets_root={assets_dir}):\n" + "\n".join(missing))

    _clear_scene()
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE" if args.engine == "eevee" else "CYCLES"
    scene.render.fps = args.fps
    total_frames = int(round(args.duration * args.fps))
    scene.frame_start = 1
    scene.frame_end = total_frames
    try:
        width_str, height_str = args.res.lower().split("x", 1)
        scene.render.resolution_x = int(width_str)
        scene.render.resolution_y = int(height_str)
    except ValueError:
        scene.render.resolution_x = 1280
        scene.render.resolution_y = 720
    scene.render.image_settings.file_format = "PNG"
    frames_dir = output_path.parent / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(frames_dir / "frame_%05d")

    scene.render.use_freestyle = args.outline_mode == "freestyle"
    if hasattr(scene.render, "line_thickness"):
        scene.render.line_thickness = 1.5
    if hasattr(scene, "eevee"):
        _configure_eevee(scene, args.quality)

    objects = _create_scene(assets_dir, args.asset_mode)
    _add_vfx(
        assets_dir,
        scene,
        objects.get("camera"),
        scene.frame_end,
        args.vfx_emission_strength,
        args.vfx_scale,
        args.vfx_screen_coverage,
    )
    print("[POSTFX] enabled=", args.postfx)
    if args.postfx == "on":
        _setup_compositor(scene)
    envelope = _load_rms_envelope(Path(args.audio) if args.audio else Path(), args.fps, scene.frame_end)
    mouth_keyframes = _animate(objects, envelope, args.fps, assets_dir, args.asset_mode)

    bpy.ops.render.render(animation=True, write_still=False)

    report_path.write_text(
        json.dumps(
            {
                "mouth_keyframes": mouth_keyframes,
                "frame_end": scene.frame_end,
                "fps": scene.render.fps,
                "vfx_emission_strength": args.vfx_emission_strength,
                "vfx_scale": args.vfx_scale,
                "vfx_screen_coverage": args.vfx_screen_coverage,
                "parsed_args": vars(args),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
