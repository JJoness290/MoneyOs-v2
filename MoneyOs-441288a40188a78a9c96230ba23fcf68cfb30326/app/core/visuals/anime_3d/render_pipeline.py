from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import wave
import zlib
from typing import Callable

from moviepy.editor import AudioFileClip, CompositeAudioClip

from app.config import (
    ANIME3D_ASSET_MODE,
    ANIME3D_FPS,
    ANIME3D_QUALITY,
    ANIME3D_RESOLUTION,
    ANIME3D_SECONDS,
    ANIME3D_STYLE_PRESET,
    ANIME3D_OUTLINE_MODE,
    ANIME3D_POSTFX,
    OUTPUT_DIR,
    VFX_EMISSION_STRENGTH,
    VFX_SCALE,
    VFX_SCREEN_COVERAGE,
)
from app.core.paths import get_assets_root
from app.core.tts import generate_tts
from app.core.visuals.anime_3d.blender_runner import build_blender_command
from src.utils.cli_args import add_opt, validate_no_empty_value_flags
from app.core.visuals.anime_3d.validators import validate_episode
from app.core.visuals.ffmpeg_utils import has_nvenc, run_ffmpeg
from src.utils.win_paths import planned_paths_preflight


@dataclass(frozen=True)
class Anime3DResult:
    output_dir: Path
    final_video: Path
    audio_path: Path
    duration_seconds: float
    warnings: list[str]


StatusCallback = Callable[[dict], None] | None


def anime_3d_output_dir(job_id: str) -> Path:
    return (OUTPUT_DIR / "episodes" / job_id).resolve()


def _required_asset_paths() -> dict[str, Path]:
    assets_root = get_assets_root()
    return {
        "characters/hero.blend": assets_root / "characters" / "hero.blend",
        "characters/enemy.blend": assets_root / "characters" / "enemy.blend",
        "envs/city.blend": assets_root / "envs" / "city.blend",
        "anims/idle.fbx": assets_root / "anims" / "idle.fbx",
        "anims/run.fbx": assets_root / "anims" / "run.fbx",
        "anims/punch.fbx": assets_root / "anims" / "punch.fbx",
        "vfx/explosion.png": assets_root / "vfx" / "explosion.png",
        "vfx/energy_arc.png": assets_root / "vfx" / "energy_arc.png",
        "vfx/smoke.png": assets_root / "vfx" / "smoke.png",
    }


def _ensure_assets() -> None:
    if ANIME3D_ASSET_MODE != "local":
        return
    missing = [key for key, path in _required_asset_paths().items() if not path.exists()]
    if missing:
        message = f"Missing assets (assets_root={get_assets_root()}):\n" + "\n".join(
            f"- {key}" for key in missing
        )
        raise RuntimeError(message)


def _generate_base_tone(path: Path, duration_s: float, sample_rate: int = 44100) -> None:
    total_frames = int(duration_s * sample_rate)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        for i in range(total_frames):
            t = i / sample_rate
            mod = 0.5 + 0.5 * math.sin(2 * math.pi * 0.5 * t)
            sample = int(12000 * mod * math.sin(2 * math.pi * 220 * t))
            handle.writeframes(sample.to_bytes(2, byteorder="little", signed=True))


def _emit_status(
    status_callback: StatusCallback,
    *,
    stage_key: str,
    status: str,
    progress_pct: int | None = None,
    extra: dict | None = None,
) -> None:
    if not status_callback:
        return
    payload = {
        "stage_key": stage_key,
        "status": status,
        "progress_pct": progress_pct,
        "extra": extra,
    }
    status_callback(payload)


def _finalize_mux(video_path: Path, audio_path: Path, output_path: Path) -> None:
    args = ["ffmpeg", "-y", "-i", str(video_path)]
    if not audio_path.exists() or audio_path.stat().st_size == 0:
        raise RuntimeError(f"audio missing or empty during mux: {audio_path}")
    args += ["-i", str(audio_path), "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest"]
    args.append(str(output_path))
    print("[ENC] ffmpeg mux:", " ".join(args))
    run_ffmpeg(args)


def _update_report_warnings(report_path: Path, warnings: list[str]) -> None:
    if not warnings or not report_path.exists():
        return
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        payload = {}
    payload["warnings"] = sorted(set(payload.get("warnings", []) + warnings))
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    encode_report_path = report_path.with_name("encode_report.json")
    if encode_report_path.exists():
        try:
            encode_payload = json.loads(encode_report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            encode_payload = {}
        encode_payload["warnings"] = sorted(set(encode_payload.get("warnings", []) + warnings))
        encode_report_path.write_text(json.dumps(encode_payload, indent=2), encoding="utf-8")


def _assemble_frames_video(
    frames_dir: Path,
    fps: int,
    audio_path: Path,
    output_path: Path,
    warnings: list[str],
    report_path: Path | None = None,
) -> None:
    _ = audio_path
    encode_report_path = output_path.with_name("encode_report.json")
    frame_files = sorted(frames_dir.glob("frame_*.png"))
    if not frame_files:
        raise RuntimeError(f"No frames found in {frames_dir}")
    first_frame = frames_dir / "frame_0001.png"
    if not first_frame.exists():
        sample = [path.name for path in frame_files[:10]]
        raise RuntimeError(
            "Missing expected first frame frame_0001.png. "
            f"Sample frames: {sample}"
        )
    pattern_regex = re.compile(r"frame_(\d+)\.png$")
    numbers = []
    pad_width = None
    for frame in frame_files:
        match = pattern_regex.match(frame.name)
        if not match:
            continue
        number_str = match.group(1)
        if pad_width is None:
            pad_width = len(number_str)
        numbers.append(int(number_str))
    if not numbers:
        raise RuntimeError(f"No frame sequence detected in {frames_dir}")
    start_number = min(numbers)
    expected = list(range(start_number, start_number + len(numbers)))
    if numbers != expected:
        sample = numbers[:10]
        raise RuntimeError(f"Frame sequence has gaps. Sample numbers: {sample}")
    pad_width = pad_width or 4
    pattern = f"frame_%0{pad_width}d.png"
    frame_count = len(numbers)
    use_gpu = os.getenv("MONEYOS_USE_GPU", "0") == "1"
    use_nvenc = use_gpu and has_nvenc()
    print(f"[ENC] frames={frame_count} encoder={'h264_nvenc' if use_nvenc else 'libx264'}")
    args = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-start_number",
        str(start_number),
        "-i",
        str(frames_dir / pattern),
    ]
    if use_nvenc:
        args += [
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p7",
            "-cq",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ]
        encoder_name = "h264_nvenc"
    else:
        args += [
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ]
        encoder_name = "libx264"
    args.append(str(output_path))
    print("[ENC] ffmpeg:", " ".join(args))
    encode_report = {
        "pattern": pattern,
        "start_number": start_number,
        "pad_width": pad_width,
        "encoder": encoder_name,
        "ffmpeg_args": args,
        "warnings": warnings,
    }
    try:
        run_ffmpeg(args)
    except Exception as exc:  # noqa: BLE001
        encode_report["error"] = str(exc)
        encode_report_path.write_text(json.dumps(encode_report, indent=2), encoding="utf-8")
        raise
    _augment_encode_report(
        encode_report_path,
        encode_report,
        report_path,
        frames_dir,
    )


def _augment_encode_report(
    encode_report_path: Path,
    encode_report: dict,
    report_path: Path | None,
    frames_dir: Path,
) -> None:
    if report_path and report_path.exists():
        try:
            render_payload = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            render_payload = {}
        for key in ("seed", "fingerprint"):
            if key in render_payload:
                encode_report[key] = render_payload[key]
    if shutil.which("ffmpeg"):
        first_frame = frames_dir / "frame_0001.png"
        last_frame = frames_dir / "frame_0001.png"
        if first_frame.exists():
            encode_report["first_frame_hash"] = _framehash_for_image(first_frame)
        frame_files = sorted(frames_dir.glob("frame_*.png"))
        if frame_files:
            last_frame = frame_files[-1]
        if last_frame.exists():
            encode_report["last_frame_hash"] = _framehash_for_image(last_frame)
    encode_report_path.write_text(json.dumps(encode_report, indent=2), encoding="utf-8")


def _framehash_for_image(image_path: Path) -> str | None:
    result = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(image_path),
            "-frames:v",
            "1",
            "-pix_fmt",
            "rgb24",
            "-f",
            "framehash",
            "-",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    lines = [line for line in result.stdout.splitlines() if line and not line.startswith("#")]
    if not lines:
        return None
    return lines[-1].split(",")[-1].strip()


def _validate_blender_artifacts(output_dir: Path, report_path: Path) -> None:
    blender_cmd_path = output_dir / "blender_cmd.txt"
    if not blender_cmd_path.exists():
        raise RuntimeError("blender_cmd.txt missing after render")
    cmd_text = blender_cmd_path.read_text(encoding="utf-8")
    if "--seed" not in cmd_text:
        raise RuntimeError("blender_cmd.txt missing --seed argument")
    if "--fingerprint" not in cmd_text:
        raise RuntimeError("blender_cmd.txt missing --fingerprint argument")
    if not report_path.exists():
        raise RuntimeError("render_report.json missing after render")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError("render_report.json unreadable") from exc
    if not report.get("seed") or not report.get("fingerprint"):
        raise RuntimeError("render_report.json missing seed/fingerprint")


def _derive_episode_seed(job_id: str | None, output_dir: Path) -> int:
    seed_source = job_id if job_id else str(output_dir)
    seed_value = zlib.crc32(seed_source.encode("utf-8")) & 0xFFFFFFFF
    if seed_value == 0:
        seed_value = 1
    return int(min(seed_value, 2_000_000_000))


def _build_fingerprint(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:12]


def _generate_audio(output_dir: Path, duration_s: float, status_callback: StatusCallback) -> Path:
    _emit_status(status_callback, stage_key="audio", status="Generating audio", progress_pct=8)
    base_tone = output_dir / "base_tone.wav"
    _generate_base_tone(base_tone, duration_s)
    tts_path = output_dir / "tts.wav"
    try:
        generate_tts("MoneyOS 3D test line. Audio and lips are synced.", tts_path)
    except Exception:
        tts_path = None
    final_path = output_dir / "audio.wav"
    base_clip = AudioFileClip(str(base_tone))
    clips = [base_clip.volumex(0.35)]
    if tts_path and tts_path.exists():
        clips.append(AudioFileClip(str(tts_path)).volumex(1.0).set_start(1.0))
    composite = CompositeAudioClip(clips).set_duration(duration_s)
    composite.write_audiofile(str(final_path), fps=44100, logger=None)
    composite.close()
    for clip in clips:
        clip.close()
    _emit_status(status_callback, stage_key="audio", status="Generating audio", progress_pct=12)
    return final_path


def render_anime_3d_60s(
    job_id: str,
    status_callback: StatusCallback = None,
    overrides: dict | None = None,
) -> Anime3DResult:
    warnings: list[str] = []
    render_preset = os.getenv("MONEYOS_RENDER_PRESET", "fast_proof").strip().lower()
    if render_preset not in {"fast_proof", "phase15_quality"}:
        render_preset = "fast_proof"
    env_template = os.getenv("MONEYOS_ENV_TEMPLATE", "room").strip().lower()
    fast_proof = render_preset == "fast_proof"
    phase15 = render_preset == "phase15_quality"
    try:
        phase15_samples = int(os.getenv("MONEYOS_PHASE15_SAMPLES", "128"))
    except ValueError:
        phase15_samples = 128
    try:
        phase15_bounces = int(os.getenv("MONEYOS_PHASE15_BOUNCES", "6"))
    except ValueError:
        phase15_bounces = 6
    try:
        phase15_tile = int(os.getenv("MONEYOS_PHASE15_TILE", "256"))
    except ValueError:
        phase15_tile = 256
    phase15_res = os.getenv("MONEYOS_PHASE15_RES", "1920x1080")
    duration_s = float(ANIME3D_SECONDS)
    fps = ANIME3D_FPS
    res = f"{ANIME3D_RESOLUTION[0]}x{ANIME3D_RESOLUTION[1]}"
    postfx = "on" if ANIME3D_POSTFX else "off"
    outline_mode = ANIME3D_OUTLINE_MODE
    quality = ANIME3D_QUALITY
    style_preset = ANIME3D_STYLE_PRESET
    vfx_emission_strength = VFX_EMISSION_STRENGTH
    vfx_scale = VFX_SCALE
    vfx_screen_coverage = VFX_SCREEN_COVERAGE
    environment = env_template
    character_asset = None
    mode = "default"
    seed_value: int | None = None
    strict_assets = 1
    action = None
    camera_preset = None
    start_frame = None
    disable_overlays = True
    overrides = overrides or {}
    if overrides.get("render_preset"):
        render_preset = str(overrides["render_preset"]).strip().lower()
        if render_preset not in {"fast_proof", "phase15_quality"}:
            render_preset = "fast_proof"
        fast_proof = render_preset == "fast_proof"
        phase15 = render_preset == "phase15_quality"
    if overrides.get("environment"):
        environment = str(overrides["environment"]).strip().lower()
    if overrides.get("character_asset"):
        character_asset = str(overrides["character_asset"])
    if overrides.get("mode"):
        mode = str(overrides["mode"]).strip().lower()
    if overrides.get("seed") is not None:
        seed_value = int(overrides["seed"])
    if overrides.get("strict_assets") is not None:
        strict_assets = int(bool(overrides["strict_assets"]))
    else:
        strict_assets = 1
    if overrides.get("action"):
        action = str(overrides["action"]).strip().lower()
    if overrides.get("camera_preset"):
        camera_preset = str(overrides["camera_preset"]).strip().lower()
    if overrides.get("start_frame") is not None:
        start_frame = int(overrides["start_frame"])
    if overrides.get("disable_overlays") is not None:
        disable_overlays = bool(overrides["disable_overlays"])
    if overrides.get("duration_seconds") is not None:
        duration_s = float(overrides["duration_seconds"])
    if overrides.get("fps") is not None:
        fps = int(overrides["fps"])
    if overrides.get("res"):
        res = str(overrides["res"])
    if overrides.get("quality"):
        quality = str(overrides["quality"])
    if overrides.get("style_preset"):
        style_preset = str(overrides["style_preset"])
    if overrides.get("outline_mode"):
        outline_mode = str(overrides["outline_mode"])
    if overrides.get("postfx") is not None:
        postfx = "on" if bool(overrides["postfx"]) else "off"
    if disable_overlays:
        postfx = "off"
    if overrides.get("vfx_emission_strength") is not None:
        vfx_emission_strength = float(overrides["vfx_emission_strength"])
    if overrides.get("vfx_scale") is not None:
        vfx_scale = float(overrides["vfx_scale"])
    if overrides.get("vfx_screen_coverage") is not None:
        vfx_screen_coverage = float(overrides["vfx_screen_coverage"])
    if phase15 and "res" not in overrides:
        res = phase15_res
    if phase15 and "fps" not in overrides:
        fps = 30
    if fast_proof:
        res = "1280x720"
        postfx = "off"
        outline_mode = "off"
        vfx_emission_strength = 0.0
        quality = "fast"
    if duration_s <= 0:
        raise RuntimeError("Duration must be provided from audio beats and be > 0 seconds.")
    _ensure_assets()
    output_dir = anime_3d_output_dir(job_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    if seed_value is None:
        seed_value = _derive_episode_seed(job_id, output_dir)
    fingerprint_payload = {
        "episode_id": job_id,
        "seed": seed_value,
        "assets_dir": str(get_assets_root()),
        "environment": environment,
        "mode": mode,
        "style_preset": style_preset,
        "duration": f"{duration_s:.6f}",
        "fps": fps,
        "res": res,
    }
    fingerprint = _build_fingerprint(fingerprint_payload)
    planned_paths = [
        output_dir / "render_report.json",
        output_dir / "segment.mp4",
        output_dir / "final.mp4",
        output_dir / "frames" / "frame_0001.png",
        output_dir / "blender_cmd.txt",
    ]
    ok, longest_path, longest_len = planned_paths_preflight(planned_paths)
    if not ok:
        short_root = os.getenv("MONEYOS_SHORT_WORKDIR", "C:\\MoneyOS\\work")
        raise RuntimeError(
            "Path too long (WinError 206). "
            f"Using short workdir: {short_root}. "
            f"Longest path: {longest_path} ({longest_len})."
        )
    audio_path = _generate_audio(output_dir, duration_s, status_callback)
    video_path = output_dir / "segment.mp4"
    video_raw_path = output_dir / "video_raw.mp4"
    report_path = output_dir / "render_report.json"
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__).parent / "blender" / "render_segment.py"
    script_copy_path = output_dir / "blender_script.py"
    script_copy_path.write_text(script_path.read_text(encoding="utf-8"), encoding="utf-8")
    blender_args: list[str] = []
    add_opt(blender_args, "--output", video_path)
    add_opt(blender_args, "--audio", audio_path)
    add_opt(blender_args, "--report", report_path)
    add_opt(blender_args, "--assets-dir", get_assets_root())
    add_opt(blender_args, "--asset-mode", ANIME3D_ASSET_MODE)
    add_opt(blender_args, "--strict-assets", strict_assets)
    if phase15:
        add_opt(blender_args, "--engine", "cycles")
    add_opt(blender_args, "--render-preset", render_preset)
    add_opt(blender_args, "--environment", environment)
    add_opt(blender_args, "--character-asset", character_asset)
    add_opt(blender_args, "--mode", mode)
    add_opt(blender_args, "--seed", seed_value)
    add_opt(blender_args, "--fingerprint", fingerprint)
    add_opt(blender_args, "--action", action)
    add_opt(blender_args, "--camera-preset", camera_preset)
    add_opt(blender_args, "--start-frame", start_frame)
    add_opt(blender_args, "--style-preset", style_preset)
    add_opt(blender_args, "--outline-mode", outline_mode)
    add_opt(blender_args, "--postfx", postfx)
    add_opt(blender_args, "--quality", quality)
    add_opt(blender_args, "--phase15-samples", phase15_samples)
    add_opt(blender_args, "--phase15-bounces", phase15_bounces)
    add_opt(blender_args, "--phase15-res", res)
    add_opt(blender_args, "--phase15-tile", phase15_tile)
    add_opt(blender_args, "--res", res)
    add_opt(blender_args, "--duration", f"{duration_s:.6f}")
    add_opt(blender_args, "--fps", fps)
    add_opt(blender_args, "--vfx-emission-strength", vfx_emission_strength)
    add_opt(blender_args, "--vfx-scale", vfx_scale)
    add_opt(blender_args, "--vfx-screen-coverage", vfx_screen_coverage)
    duration_label = f"{duration_s:.6f}s"
    frame_count = math.ceil(duration_s * fps)
    _emit_status(
        status_callback,
        stage_key="blender",
        status=f"[DURATION] source=audio beat={duration_label} fps={fps} frames={frame_count}",
        progress_pct=14,
    )
    validate_no_empty_value_flags(
        blender_args,
        {
            "--output",
            "--audio",
            "--report",
            "--assets-dir",
            "--asset-mode",
            "--strict-assets",
            "--engine",
            "--render-preset",
            "--environment",
            "--character-asset",
            "--mode",
            "--seed",
            "--fingerprint",
            "--style-preset",
            "--outline-mode",
            "--postfx",
            "--quality",
            "--phase15-samples",
            "--phase15-bounces",
            "--phase15-res",
            "--phase15-tile",
            "--res",
            "--duration",
            "--fps",
            "--vfx-emission-strength",
            "--vfx-scale",
            "--vfx-screen-coverage",
        },
    )
    cmd = build_blender_command(script_copy_path, blender_args)
    blender_cmd_path = output_dir / "blender_cmd.txt"
    blender_stdout_path = output_dir / "blender_stdout.txt"
    blender_stderr_path = output_dir / "blender_stderr.txt"
    blender_cmd_path.write_text(" ".join(cmd), encoding="utf-8")
    character_asset_label = character_asset if character_asset else "<omitted>"
    _emit_status(
        status_callback,
        stage_key="blender",
        status=f"[BLENDER] character_asset: {character_asset_label}",
        progress_pct=15,
    )

    _emit_status(status_callback, stage_key="blender", status="Launching Blender", progress_pct=15)
    with blender_stdout_path.open("w", encoding="utf-8") as stdout_handle, blender_stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_handle:
        process = subprocess.Popen(
            cmd,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
        )
        total_frames = max(1, int(math.ceil(duration_s * fps)))
        last_update = 0.0
        while process.poll() is None:
            now = time.time()
            if now - last_update >= 2.0:
                frame_count = len(list(frames_dir.glob("frame_*.png")))
                progress = 10 + int(min(frame_count / total_frames, 1.0) * 84)
                _emit_status(
                    status_callback,
                    stage_key="frames",
                    status="Rendering frames",
                    progress_pct=progress,
                    extra={"frames_rendered": frame_count, "total_frames": total_frames},
                )
                last_update = now
            time.sleep(0.2)
        returncode = process.wait()
    stdout_text = blender_stdout_path.read_text(encoding="utf-8") if blender_stdout_path.exists() else ""
    stderr_text = blender_stderr_path.read_text(encoding="utf-8") if blender_stderr_path.exists() else ""
    if returncode != 0:
        tail_stdout = stdout_text[-2000:]
        tail_stderr = stderr_text[-2000:]
        raise RuntimeError(
            "Blender render failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"Stdout (tail):\n{tail_stdout}\n"
            f"Stderr (tail):\n{tail_stderr}"
        )
    frame_list = sorted(frames_dir.glob("frame_*.png"))
    if not frame_list:
        frame_list = sorted(frames_dir.glob("*.png"))
    if len(frame_list) < 2:
        contents = "\n".join(path.name for path in list(frames_dir.glob("*"))[:200])
        tail_stdout = stdout_text[-2000:]
        tail_stderr = stderr_text[-2000:]
        raise RuntimeError(
            "No frames rendered.\n"
            f"Command: {' '.join(cmd)}\n"
            f"Frames dir: {frames_dir}\n"
            f"Contents:\n{contents}\n"
            f"Stdout (tail):\n{tail_stdout}\n"
            f"Stderr (tail):\n{tail_stderr}"
        )
    _emit_status(status_callback, stage_key="encode", status="Encoding video", progress_pct=95)
    _assemble_frames_video(
        frames_dir,
        fps,
        audio_path,
        video_path,
        warnings,
        report_path,
    )
    if not video_path.exists() and video_raw_path.exists():
        video_path = video_raw_path
    if not video_path.exists():
        raise RuntimeError("segment.mp4 missing after frame encode")
    final_path = output_dir / "final.mp4"
    _finalize_mux(video_path, audio_path, final_path)
    _validate_blender_artifacts(output_dir, report_path)
    validation = validate_episode(final_path, audio_path, report_path)
    warnings.extend(validation.warnings)
    if not validation.valid:
        warnings.append(validation.message)
        _update_report_warnings(report_path, warnings)
        if not fast_proof:
            raise RuntimeError(validation.message)
    _update_report_warnings(report_path, warnings)
    return Anime3DResult(
        output_dir=output_dir,
        final_video=final_path,
        audio_path=audio_path,
        duration_seconds=duration_s,
        warnings=warnings,
    )


def finalize_anime_3d(job_id: str, status_callback: StatusCallback = None) -> Anime3DResult:
    warnings: list[str] = []
    output_dir = anime_3d_output_dir(job_id)
    audio_path = output_dir / "audio.wav"
    frames_dir = output_dir / "frames"
    video_path = output_dir / "segment.mp4"
    video_raw_path = output_dir / "video_raw.mp4"
    final_path = output_dir / "final.mp4"
    report_path = output_dir / "render_report.json"
    if not video_path.exists() and frames_dir.exists():
        _emit_status(status_callback, stage_key="encode", status="Encoding video", progress_pct=95)
        _assemble_frames_video(frames_dir, ANIME3D_FPS, audio_path, video_path, warnings, report_path)
    if not video_path.exists() and video_raw_path.exists():
        video_path = video_raw_path
    if not video_path.exists():
        raise RuntimeError("No video_raw.mp4 or segment.mp4 found to finalize")
    _finalize_mux(video_path, audio_path, final_path)
    _validate_blender_artifacts(output_dir, report_path)
    validation = validate_episode(final_path, audio_path, report_path)
    warnings.extend(validation.warnings)
    if not validation.valid:
        warnings.append(validation.message)
        _update_report_warnings(report_path, warnings)
        raise RuntimeError(validation.message)
    _update_report_warnings(report_path, warnings)
    return Anime3DResult(
        output_dir=output_dir,
        final_video=final_path,
        audio_path=audio_path,
        duration_seconds=float(ANIME3D_SECONDS),
        warnings=warnings,
    )
