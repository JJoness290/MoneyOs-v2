from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import wave

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
    ASSETS_DIR,
    OUTPUT_DIR,
)
from app.core.tts import generate_tts
from app.core.visuals.anime_3d.blender_runner import BlenderCommand, run_blender_capture
from app.core.visuals.anime_3d.validators import validate_episode
from app.core.visuals.ffmpeg_utils import run_ffmpeg, select_video_encoder


@dataclass(frozen=True)
class Anime3DResult:
    output_dir: Path
    final_video: Path
    audio_path: Path
    duration_seconds: float


def anime_3d_output_dir(job_id: str) -> Path:
    return (OUTPUT_DIR / "episodes" / job_id).resolve()


def _required_asset_paths() -> dict[str, Path]:
    return {
        "characters/hero.blend": ASSETS_DIR / "characters" / "hero.blend",
        "characters/enemy.blend": ASSETS_DIR / "characters" / "enemy.blend",
        "envs/city.blend": ASSETS_DIR / "envs" / "city.blend",
        "anims/idle.fbx": ASSETS_DIR / "anims" / "idle.fbx",
        "anims/run.fbx": ASSETS_DIR / "anims" / "run.fbx",
        "anims/punch.fbx": ASSETS_DIR / "anims" / "punch.fbx",
        "vfx/explosion.png": ASSETS_DIR / "vfx" / "explosion.png",
        "vfx/energy_arc.png": ASSETS_DIR / "vfx" / "energy_arc.png",
        "vfx/smoke.png": ASSETS_DIR / "vfx" / "smoke.png",
    }


def _ensure_assets() -> None:
    if ANIME3D_ASSET_MODE != "local":
        return
    missing = [key for key, path in _required_asset_paths().items() if not path.exists()]
    if missing:
        message = "Missing assets:\n" + "\n".join(f"- {key}" for key in missing)
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


def _generate_audio(output_dir: Path, duration_s: float) -> Path:
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
    return final_path


def render_anime_3d_60s(job_id: str) -> Anime3DResult:
    duration_s = float(ANIME3D_SECONDS)
    _ensure_assets()
    output_dir = anime_3d_output_dir(job_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_path = _generate_audio(output_dir, duration_s)
    video_path = output_dir / "segment.mp4"
    report_path = output_dir / "render_report.json"
    frames_dir = output_dir / "frames"
    script_path = Path(__file__).parent / "blender" / "render_segment.py"
    result = run_blender_capture(
        BlenderCommand(
            script_path=script_path,
            args=[
                "--output",
                str(video_path),
                "--audio",
                str(audio_path),
                "--report",
                str(report_path),
                "--assets-dir",
                str(ASSETS_DIR),
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
            ],
        )
    )
    (output_dir / "blender_stdout.txt").write_text(result.stdout or "", encoding="utf-8")
    (output_dir / "blender_stderr.txt").write_text(result.stderr or "", encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"Blender render failed: {result.stderr.strip()}")
    frame_list = sorted(frames_dir.glob("frame_*.png"))
    if not frame_list:
        contents = "\n".join(path.name for path in frames_dir.glob("*"))
        raise RuntimeError(f"No frames rendered in {frames_dir}. Contents:\n{contents}")
    encode_args, _ = select_video_encoder()
    frame_pattern = str(frames_dir / "frame_%04d.png")
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
