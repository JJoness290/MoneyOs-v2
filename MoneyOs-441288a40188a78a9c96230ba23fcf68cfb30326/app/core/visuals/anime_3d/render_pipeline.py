from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import subprocess
import time
import wave
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
)
from app.core.paths import get_assets_root
from app.core.tts import generate_tts
from app.core.visuals.anime_3d.blender_runner import build_blender_command
from app.core.visuals.anime_3d.validators import validate_episode
from app.core.visuals.ffmpeg_utils import run_ffmpeg, select_video_encoder


@dataclass(frozen=True)
class Anime3DResult:
    output_dir: Path
    final_video: Path
    audio_path: Path
    duration_seconds: float


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


def render_anime_3d_60s(job_id: str, status_callback: StatusCallback = None) -> Anime3DResult:
    duration_s = float(ANIME3D_SECONDS)
    _ensure_assets()
    output_dir = anime_3d_output_dir(job_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_path = _generate_audio(output_dir, duration_s, status_callback)
    video_path = output_dir / "segment.mp4"
    report_path = output_dir / "render_report.json"
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__).parent / "blender" / "render_segment.py"
    script_copy_path = output_dir / "blender_script.py"
    script_copy_path.write_text(script_path.read_text(encoding="utf-8"), encoding="utf-8")
    blender_args = [
        "--output",
        str(video_path),
        "--audio",
        str(audio_path),
        "--report",
        str(report_path),
        "--assets-dir",
        str(get_assets_root()),
        "--asset-mode",
        ANIME3D_ASSET_MODE,
        "--style-preset",
        ANIME3D_STYLE_PRESET,
        "--outline-mode",
        ANIME3D_OUTLINE_MODE,
        "--postfx",
        "on" if ANIME3D_POSTFX else "off",
        "--quality",
        ANIME3D_QUALITY,
        "--res",
        f"{ANIME3D_RESOLUTION[0]}x{ANIME3D_RESOLUTION[1]}",
        "--duration",
        f"{duration_s:.2f}",
        "--fps",
        str(ANIME3D_FPS),
    ]
    cmd = build_blender_command(script_copy_path, blender_args)
    blender_cmd_path = output_dir / "blender_cmd.txt"
    blender_stdout_path = output_dir / "blender_stdout.txt"
    blender_stderr_path = output_dir / "blender_stderr.txt"
    blender_cmd_path.write_text(" ".join(cmd), encoding="utf-8")

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
        total_frames = max(1, int(round(duration_s * ANIME3D_FPS)))
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
    encode_args, _ = select_video_encoder()
    frame_pattern = str(frames_dir / "frame_%06d.png")
    args = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(ANIME3D_FPS),
        "-i",
        frame_pattern,
        *encode_args,
        str(video_path),
    ]
    run_ffmpeg(args)
    if not video_path.exists():
        raise RuntimeError("segment.mp4 missing after frame encode")
    final_path = output_dir / "final.mp4"
    _emit_status(status_callback, stage_key="mux", status="Muxing audio", progress_pct=98)
    args = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        *encode_args,
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        str(final_path),
    ]
    run_ffmpeg(args)
    validation = validate_episode(final_path, audio_path, report_path)
    if not validation.valid:
        raise RuntimeError(validation.message)
    return Anime3DResult(
        output_dir=output_dir,
        final_video=final_path,
        audio_path=audio_path,
        duration_seconds=duration_s,
    )
