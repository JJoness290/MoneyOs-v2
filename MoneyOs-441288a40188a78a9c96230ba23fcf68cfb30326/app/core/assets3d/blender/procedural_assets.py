from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import random
import sys

import bpy


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-dir", required=True)
    parser.add_argument("--engine", default="")
    parser.add_argument("--gpu", default="0")
    args = sys.argv
    if "--" in args:
        args = args[args.index("--") + 1 :]
    else:
        args = []
    return parser.parse_args(args)


def _reset_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for collection in bpy.data.collections:
        if collection.name != "Collection":
            bpy.data.collections.remove(collection)
    bpy.ops.outliner.orphans_purge(do_recursive=True)


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _create_material(name: str, color: tuple[float, float, float, float]) -> bpy.types.Material:
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    principled = nodes.get("Principled BSDF")
    if principled:
        principled.inputs["Base Color"].default_value = color
        principled.inputs["Roughness"].default_value = 0.5
    return mat


def _create_city(assets_dir: Path) -> None:
    _reset_scene()
    bpy.ops.mesh.primitive_plane_add(size=40, location=(0, 0, 0))
    ground = bpy.context.active_object
    ground.name = "Ground"
    ground_mat = _create_material("GroundMat", (0.05, 0.05, 0.05, 1))
    ground.data.materials.append(ground_mat)

    rng = random.Random(123)
    for idx in range(18):
        x = rng.uniform(-15, 15)
        y = rng.uniform(-15, 15)
        height = rng.uniform(3, 12)
        bpy.ops.mesh.primitive_cube_add(size=1, location=(x, y, height / 2))
        building = bpy.context.active_object
        building.scale = (rng.uniform(1.5, 3.5), rng.uniform(1.5, 3.5), height)
        mat = _create_material(f"BuildingMat{idx}", (0.1, 0.1, 0.12, 1))
        mat.node_tree.nodes["Principled BSDF"].inputs["Emission"].default_value = (
            0.8,
            0.6,
            0.2,
            1,
        )
        mat.node_tree.nodes["Principled BSDF"].inputs["Emission Strength"].default_value = (
            rng.uniform(0.3, 0.8)
        )
        building.data.materials.append(mat)

    bpy.ops.object.light_add(type="AREA", location=(0, 0, 20))
    light = bpy.context.active_object
    light.data.energy = 3000
    light.data.size = 10

    bpy.ops.object.camera_add(location=(18, -18, 12), rotation=(math.radians(60), 0, math.radians(45)))
    _ensure_dir(assets_dir / "envs" / "city.blend")
    bpy.ops.wm.save_as_mainfile(filepath=str(assets_dir / "envs" / "city.blend"))


def _build_armature(name: str) -> bpy.types.Object:
    bpy.ops.object.armature_add(enter_editmode=True, location=(0, 0, 1))
    armature = bpy.context.active_object
    armature.name = name
    arm = armature.data
    arm.edit_bones[0].name = "root"
    root = arm.edit_bones["root"]
    root.tail = (0, 0, 1)

    spine = arm.edit_bones.new("spine")
    spine.head = root.tail
    spine.tail = (0, 0, 1.6)
    spine.parent = root

    head = arm.edit_bones.new("head")
    head.head = spine.tail
    head.tail = (0, 0, 2.0)
    head.parent = spine

    arm_l = arm.edit_bones.new("arm.L")
    arm_l.head = (0, 0, 1.4)
    arm_l.tail = (-0.6, 0, 1.2)
    arm_l.parent = spine

    arm_r = arm.edit_bones.new("arm.R")
    arm_r.head = (0, 0, 1.4)
    arm_r.tail = (0.6, 0, 1.2)
    arm_r.parent = spine

    leg_l = arm.edit_bones.new("leg.L")
    leg_l.head = (0, 0, 1)
    leg_l.tail = (-0.3, 0, 0)
    leg_l.parent = root

    leg_r = arm.edit_bones.new("leg.R")
    leg_r.head = (0, 0, 1)
    leg_r.tail = (0.3, 0, 0)
    leg_r.parent = root

    bpy.ops.object.mode_set(mode="OBJECT")
    return armature


def _build_mesh(material: bpy.types.Material) -> bpy.types.Object:
    parts = []
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.25, location=(0, 0, 2.05))
    parts.append(bpy.context.active_object)
    bpy.ops.mesh.primitive_cylinder_add(radius=0.18, depth=1.0, location=(0, 0, 1.3))
    parts.append(bpy.context.active_object)
    bpy.ops.mesh.primitive_cylinder_add(radius=0.09, depth=1.0, location=(-0.6, 0, 1.2))
    parts.append(bpy.context.active_object)
    bpy.ops.mesh.primitive_cylinder_add(radius=0.09, depth=1.0, location=(0.6, 0, 1.2))
    parts.append(bpy.context.active_object)
    bpy.ops.mesh.primitive_cylinder_add(radius=0.1, depth=1.0, location=(-0.3, 0, 0.5))
    parts.append(bpy.context.active_object)
    bpy.ops.mesh.primitive_cylinder_add(radius=0.1, depth=1.0, location=(0.3, 0, 0.5))
    parts.append(bpy.context.active_object)
    for obj in parts:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = parts[0]
    bpy.ops.object.join()
    mesh = bpy.context.active_object
    mesh.name = "Body"
    mesh.data.materials.append(material)
    return mesh


def _create_character(assets_dir: Path, name: str, color: tuple[float, float, float, float]) -> None:
    _reset_scene()
    armature = _build_armature("Armature")
    mesh = _build_mesh(_create_material(f"{name}_Mat", color))
    mesh.select_set(True)
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.parent_set(type="ARMATURE_AUTO")
    _ensure_dir(assets_dir / "characters" / f"{name}.blend")
    bpy.ops.wm.save_as_mainfile(filepath=str(assets_dir / "characters" / f"{name}.blend"))


def _insert_pose_keyframe(armature: bpy.types.Object, frame: int, rotations: dict[str, tuple[float, float, float]]):
    bpy.context.scene.frame_set(frame)
    for bone_name, rot in rotations.items():
        pose_bone = armature.pose.bones.get(bone_name)
        if not pose_bone:
            continue
        pose_bone.rotation_mode = "XYZ"
        pose_bone.rotation_euler = rot
        pose_bone.keyframe_insert(data_path="rotation_euler", frame=frame)


def _export_animation(
    assets_dir: Path,
    action_name: str,
    keyframes: list[tuple[int, dict[str, tuple[float, float, float]]]],
    end_frame: int,
) -> None:
    _reset_scene()
    armature = _build_armature("Armature")
    mesh = _build_mesh(_create_material("AnimMat", (0.7, 0.7, 0.7, 1)))
    mesh.select_set(True)
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.parent_set(type="ARMATURE_AUTO")
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = end_frame
    action = bpy.data.actions.new(name=action_name)
    armature.animation_data_create()
    armature.animation_data.action = action
    for frame, rotations in keyframes:
        _insert_pose_keyframe(armature, frame, rotations)
    bpy.ops.object.select_all(action="DESELECT")
    armature.select_set(True)
    mesh.select_set(True)
    bpy.context.view_layer.objects.active = armature
    _ensure_dir(assets_dir / "anims" / f"{action_name}.fbx")
    bpy.ops.export_scene.fbx(
        filepath=str(assets_dir / "anims" / f"{action_name}.fbx"),
        use_selection=True,
        bake_anim=True,
        bake_anim_use_all_actions=False,
        bake_anim_force_startend_keying=True,
        add_leaf_bones=False,
        object_types={"ARMATURE", "MESH"},
    )


def main() -> None:
    args = _parse_args()
    assets_dir = Path(args.assets_dir)
    os.environ["MONEYOS_ASSETS_DIR"] = str(assets_dir)
    _create_city(assets_dir)
    _create_character(assets_dir, "hero", (0.2, 0.5, 1.0, 1))
    _create_character(assets_dir, "enemy", (1.0, 0.2, 0.2, 1))
    idle_frames = [
        (1, {"arm.L": (0.0, 0.0, 0.1), "arm.R": (0.0, 0.0, -0.1)}),
        (15, {"arm.L": (0.05, 0.0, 0.15), "arm.R": (0.05, 0.0, -0.15)}),
        (30, {"arm.L": (0.0, 0.0, 0.1), "arm.R": (0.0, 0.0, -0.1)}),
    ]
    run_frames = [
        (1, {"leg.L": (0.6, 0.0, 0.0), "leg.R": (-0.6, 0.0, 0.0)}),
        (10, {"leg.L": (-0.6, 0.0, 0.0), "leg.R": (0.6, 0.0, 0.0)}),
        (20, {"leg.L": (0.6, 0.0, 0.0), "leg.R": (-0.6, 0.0, 0.0)}),
    ]
    punch_frames = [
        (1, {"arm.R": (0.0, 0.0, -0.2)}),
        (8, {"arm.R": (0.0, 0.0, -1.2)}),
        (16, {"arm.R": (0.0, 0.0, -0.2)}),
    ]
    _export_animation(assets_dir, "idle", idle_frames, 30)
    _export_animation(assets_dir, "run", run_frames, 20)
    _export_animation(assets_dir, "punch", punch_frames, 16)


if __name__ == "__main__":
    main()
