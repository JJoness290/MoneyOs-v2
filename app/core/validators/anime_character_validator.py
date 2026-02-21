from __future__ import annotations

from dataclasses import dataclass

PRIMITIVE_NAMES = ("cube", "sphere", "cylinder")


@dataclass(frozen=True)
class ValidationResult:
    armatures: int
    mesh_count: int
    vertex_count: int


def validate_anime_character_scene(scene: object, min_vertices: int = 5000) -> ValidationResult:
    objects = list(getattr(scene, "objects", []))
    armatures = [obj for obj in objects if getattr(obj, "type", "") == "ARMATURE"]
    meshes = [obj for obj in objects if getattr(obj, "type", "") == "MESH"]
    if not armatures:
        raise RuntimeError("Anime character validation failed: missing armature")

    character_meshes = [m for m in meshes if m.get("mo_role") == "subject" or "vrm" in m.name.lower()]
    if not character_meshes:
        character_meshes = meshes

    for mesh in character_meshes:
        lowered = mesh.name.lower()
        if any(name in lowered for name in PRIMITIVE_NAMES):
            raise RuntimeError(f"Anime character validation failed: primitive mesh detected ({mesh.name})")

    vertex_count = 0
    for mesh in character_meshes:
        data = getattr(mesh, "data", None)
        vertices = getattr(data, "vertices", []) if data else []
        vertex_count += len(vertices)

    if vertex_count < min_vertices:
        raise RuntimeError(
            f"Anime character validation failed: vertex count too low ({vertex_count} < {min_vertices})"
        )

    print("ANIME_CHARACTER_VALIDATED")
    return ValidationResult(armatures=len(armatures), mesh_count=len(character_meshes), vertex_count=vertex_count)
