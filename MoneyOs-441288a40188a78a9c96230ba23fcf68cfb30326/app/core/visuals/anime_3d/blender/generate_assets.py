from __future__ import annotations

import argparse
from pathlib import Path
import math
import bpy


def _clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in bpy.data.meshes:
        bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        bpy.data.materials.remove(block)


def _create_character(name: str) -> None:
    bpy.ops.object.armature_add(enter_editmode=False, location=(0, 0, 0))
    armature = bpy.context.active_object
    armature.name = f"{name}_armature"
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.3, location=(0, 0, 1.5))
    head = bpy.context.active_object
    bpy.ops.mesh.primitive_cylinder_add(radius=0.2, depth=1.2, location=(0, 0, 0.8))
    body = bpy.context.active_object
    for obj in [head, body]:
        obj.parent = armature
    head.data.shape_keys = head.data.shape_keys or head.shape_key_add(name="Basis")
    head.shape_key_add(name="mouth_open")


def _create_environment(name: str) -> None:
    bpy.ops.mesh.primitive_plane_add(size=12.0, location=(0.0, 0.0, 0.0))
    ground = bpy.context.active_object
    ground.name = f"{name}_ground"
    for offset in (-4.0, 4.0):
        bpy.ops.mesh.primitive_plane_add(size=6.0, location=(offset, 4.0, 2.5))
        wall = bpy.context.active_object
        wall.rotation_euler.x = math.radians(90)
        wall.name = f"{name}_wall_{offset}"


def _save(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    for name in ["hero", "enemy"]:
        _clear_scene()
        _create_character(name)
        _save(output_dir / f"{name}.blend")
    for name in ["city", "street", "studio"]:
        _clear_scene()
        _create_environment(name)
        _save(output_dir / f"{name}.blend")


if __name__ == "__main__":
    main()
