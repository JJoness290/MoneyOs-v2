from __future__ import annotations

import argparse
import json
import math
import sys
import wave
from pathlib import Path

import bpy


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
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--fps", type=int, default=30)
    return parser.parse_args(argv)


def _clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in bpy.data.meshes:
        bpy.data.meshes.remove(block)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


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


def _create_scene() -> dict[str, bpy.types.Object]:
    bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, 0))
    ground = bpy.context.active_object
    bpy.ops.mesh.primitive_cube_add(size=1.2, location=(0, 0, 1))
    body = bpy.context.active_object
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.45, location=(0, 0, 2.1))
    head = bpy.context.active_object
    bpy.ops.mesh.primitive_cube_add(size=0.3, location=(0, 0.4, 1.9))
    jaw = bpy.context.active_object
    head.parent = body
    jaw.parent = body

    bpy.ops.object.light_add(type="SUN", location=(5, -5, 8))
    bpy.ops.object.camera_add(location=(6, -6, 4), rotation=(math.radians(70), 0, math.radians(45)))
    camera = bpy.context.active_object
    bpy.context.scene.camera = camera

    return {"ground": ground, "body": body, "head": head, "jaw": jaw}


def _animate(objects: dict[str, bpy.types.Object], envelope: list[float], fps: int) -> int:
    body = objects["body"]
    jaw = objects["jaw"]
    frame_count = len(envelope)
    mouth_keyframes = 0
    for frame in range(frame_count):
        bpy.context.scene.frame_set(frame + 1)
        t = frame / fps
        body.location.x = t * 0.06
        body.location.z = 1 + math.sin(t * 2.0) * 0.08
        body.keyframe_insert(data_path="location", index=-1)

        jaw.rotation_euler.x = -envelope[frame] * 0.5
        jaw.keyframe_insert(data_path="rotation_euler", index=0)
        if envelope[frame] > 0.02:
            mouth_keyframes += 1
    return mouth_keyframes


def main() -> None:
    args = _parse_args()
    output_path = Path(args.output)
    report_path = Path(args.report)
    _ensure_parent(output_path)
    _ensure_parent(report_path)
    report_path.write_text(
        json.dumps({"parsed_args": vars(args)}, indent=2),
        encoding="utf-8",
    )

    _clear_scene()
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE" if args.engine == "eevee" else "CYCLES"
    scene.render.fps = args.fps
    total_frames = int(round(args.duration * args.fps))
    scene.frame_start = 1
    scene.frame_end = total_frames
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    scene.render.image_settings.file_format = "PNG"
    frames_dir = output_path.parent / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(frames_dir / "frame_")

    objects = _create_scene()
    envelope = _load_rms_envelope(Path(args.audio) if args.audio else Path(), args.fps, scene.frame_end)
    mouth_keyframes = _animate(objects, envelope, args.fps)

    bpy.ops.render.render(animation=True, write_still=False)

    report_path.write_text(
        json.dumps(
            {
                "mouth_keyframes": mouth_keyframes,
                "frame_end": scene.frame_end,
                "fps": scene.render.fps,
                "parsed_args": vars(args),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
