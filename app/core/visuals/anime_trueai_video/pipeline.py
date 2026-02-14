from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
import subprocess
import time

from app.config import OUTPUT_DIR
from app.core.visuals.anime_trueai_video.cogvideox_provider import CogVideoXProvider
from app.core.visuals.anime_trueai_video.provider import ClipRequest, TextToVideoProvider

StatusCallback = callable


@dataclass(frozen=True)
class TrueAIPresetConfig:
    name: str
    duration_s: float
    fps: int
    width: int
    height: int
    steps: int
    guidance: float
    frames_per_clip: int


def _output_dir(job_id: str) -> Path:
    return OUTPUT_DIR / "anime_trueai_video" / job_id


def _anime_prompt(base: str, character_desc: str, shot_idx: int) -> str:
    style = (
        "Japanese anime film style, cinematic anime lighting, anime character design, "
        "sharp clean linework, vibrant colors, cel shading, dramatic shadows, expressive anime eyes, "
        "anime movie quality, high detail anime background"
    )
    return f"{style}. Character: {character_desc}. Shot {shot_idx + 1}: {base}".strip()


def _negative_prompt() -> str:
    return "photorealistic, realistic human, live action, 3D render, western cartoon, blurry, distorted, low quality, scribbles, watermark, text"


def _ffmpeg(*args: str) -> None:
    cmd = ["ffmpeg", "-y", *args]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {' '.join(cmd)} :: {proc.stderr[-1200:]}")


def _probe_duration(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {path}")
    return float(proc.stdout.strip())


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _multiple_of_8(value: int) -> int:
    value = max(64, int(value))
    return max(64, (value // 8) * 8)


def _resolve_preset(forced_preset: str | None = None) -> TrueAIPresetConfig:
    if os.getenv("MONEYOS_TRUEAI_FASTTEST", "0") == "1":
        preset = "fasttest"
    elif forced_preset:
        preset = forced_preset.strip().lower()
    else:
        preset = os.getenv("MONEYOS_TRUEAI_PRESET", "balanced").strip().lower()
    if preset not in {"fasttest", "fast", "balanced", "max"}:
        preset = "balanced"

    if preset == "fasttest":
        cfg = TrueAIPresetConfig(
            name=preset,
            duration_s=_env_float("MONEYOS_TRUEAI_DURATION_S", 12.0),
            fps=_env_int("MONEYOS_TRUEAI_FPS", 8),
            width=_env_int("MONEYOS_TRUEAI_WIDTH", 512),
            height=_env_int("MONEYOS_TRUEAI_HEIGHT", 288),
            steps=_env_int("MONEYOS_TRUEAI_STEPS", 8),
            guidance=_env_float("MONEYOS_TRUEAI_GUIDANCE", 2.8),
            frames_per_clip=_env_int("MONEYOS_TRUEAI_FRAMES_PER_CLIP", 24),
        )
    elif preset == "fast":
        cfg = TrueAIPresetConfig(
            name=preset,
            duration_s=_env_float("MONEYOS_TRUEAI_DURATION_S", 60.0),
            fps=_env_int("MONEYOS_TRUEAI_FPS", 12),
            width=_env_int("MONEYOS_TRUEAI_WIDTH", 768),
            height=_env_int("MONEYOS_TRUEAI_HEIGHT", 432),
            steps=_env_int("MONEYOS_TRUEAI_STEPS", 14),
            guidance=_env_float("MONEYOS_TRUEAI_GUIDANCE", 4.0),
            frames_per_clip=_env_int("MONEYOS_TRUEAI_FRAMES_PER_CLIP", 32),
        )
    elif preset == "max":
        cfg = TrueAIPresetConfig(
            name=preset,
            duration_s=_env_float("MONEYOS_TRUEAI_DURATION_S", 60.0),
            fps=_env_int("MONEYOS_TRUEAI_FPS", 24),
            width=_env_int("MONEYOS_TRUEAI_WIDTH", 1280),
            height=_env_int("MONEYOS_TRUEAI_HEIGHT", 720),
            steps=_env_int("MONEYOS_TRUEAI_STEPS", 36),
            guidance=_env_float("MONEYOS_TRUEAI_GUIDANCE", 6.5),
            frames_per_clip=_env_int("MONEYOS_TRUEAI_FRAMES_PER_CLIP", 48),
        )
    else:
        cfg = TrueAIPresetConfig(
            name=preset,
            duration_s=_env_float("MONEYOS_TRUEAI_DURATION_S", 60.0),
            fps=_env_int("MONEYOS_TRUEAI_FPS", 24),
            width=_env_int("MONEYOS_TRUEAI_WIDTH", 1280),
            height=_env_int("MONEYOS_TRUEAI_HEIGHT", 720),
            steps=_env_int("MONEYOS_TRUEAI_STEPS", 24),
            guidance=_env_float("MONEYOS_TRUEAI_GUIDANCE", 5.5),
            frames_per_clip=_env_int("MONEYOS_TRUEAI_FRAMES_PER_CLIP", 48),
        )

    duration_s = min(max(cfg.duration_s, 8.0 if cfg.name == "fasttest" else 2.0), 15.0 if cfg.name == "fasttest" else 120.0)
    fps = min(max(cfg.fps, 6 if cfg.name == "fasttest" else 8), 8 if cfg.name == "fasttest" else 24)
    width = _multiple_of_8(cfg.width)
    height = _multiple_of_8(cfg.height)
    steps = min(max(cfg.steps, 6 if cfg.name == "fasttest" else 8), 10 if cfg.name == "fasttest" else 50)
    guidance = min(max(cfg.guidance, 2.0), 3.5 if cfg.name == "fasttest" else 9.0)
    frames_per_clip = min(max(cfg.frames_per_clip, 8), 48)

    return TrueAIPresetConfig(
        name=cfg.name,
        duration_s=float(duration_s),
        fps=fps,
        width=width,
        height=height,
        steps=steps,
        guidance=float(guidance),
        frames_per_clip=frames_per_clip,
    )


def run_trueai_60s_job(
    job_id: str,
    prompt: str,
    status_callback=None,
    forced_preset: str | None = None,
) -> tuple[Path, Path]:
    cfg = _resolve_preset(forced_preset)
    total_seconds = cfg.duration_s
    fps = cfg.fps
    frames_per_clip = min(48, max(8, cfg.frames_per_clip))
    clip_seconds = frames_per_clip / fps
    clip_count = int(math.ceil(total_seconds / clip_seconds))
    width = cfg.width
    height = cfg.height
    steps = cfg.steps
    guidance = cfg.guidance
    seed = int(os.getenv("MONEYOS_TRUEAI_SEED", "777"))
    enable_sharpen = os.getenv("MONEYOS_POST_SHARPEN", "1" if cfg.name in {"fasttest", "fast"} else "0") == "1"

    out_dir = _output_dir(job_id)
    clips_dir = out_dir / "clips"
    final_dir = out_dir / "final"
    out_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)
    report_path = final_dir / "report.json"

    character_desc = os.getenv(
        "MONEYOS_TRUEAI_CHARACTER_DESC",
        "same anime protagonist, dark hair, school uniform, consistent face and body proportions",
    )

    provider: TextToVideoProvider = CogVideoXProvider()
    if not provider.is_available():
        raise RuntimeError("CogVideoX backend unavailable. Install diffusers/torch and model.")

    print(
        "[TRUEAI] "
        f"preset={cfg.name} duration_s={total_seconds:.3f} fps={fps} frames_per_clip={frames_per_clip} clip_seconds={clip_seconds:.3f} "
        f"clips={clip_count} steps={steps} guidance={guidance} size={width}x{height} "
        f"compile={os.getenv('MONEYOS_TRUEAI_COMPILE', '0')} compile_after_first={os.getenv('MONEYOS_TRUEAI_COMPILE_AFTER_FIRST', '1')} "
        f"post_sharpen={int(enable_sharpen)}"
    )

    generated: list[Path] = []
    started = time.time()
    for idx in range(clip_count):
        remaining_s = max(0.0, total_seconds - (idx * clip_seconds))
        target_clip_seconds = min(clip_seconds, remaining_s if remaining_s > 0 else clip_seconds)
        if status_callback:
            status_callback(f"plan → generating clip {idx + 1}/{clip_count}")
        clip_path = clips_dir / f"clip_{idx:02d}.mp4"
        request = ClipRequest(
            prompt=_anime_prompt(prompt, character_desc, idx),
            negative_prompt=_negative_prompt(),
            seed=seed,
            seconds=max(target_clip_seconds, 1.0 / fps),
            fps=fps,
            width=width,
            height=height,
            steps=steps,
            guidance=guidance,
            out_path=clip_path,
        )
        try:
            provider.generate(request)
        except Exception:
            low_clip = clips_dir / f"clip_{idx:02d}_low.mp4"
            request = ClipRequest(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                seed=request.seed,
                seconds=request.seconds,
                fps=request.fps,
                width=max(320, _multiple_of_8(width // 2)),
                height=max(192, _multiple_of_8(height // 2)),
                steps=max(6, request.steps - 2),
                guidance=max(2.0, request.guidance - 0.8),
                out_path=low_clip,
            )
            provider.generate(request)
            _ffmpeg("-i", str(low_clip), "-vf", f"scale={width}:{height}:flags=lanczos", "-r", str(fps), str(clip_path))

        clip_duration = _probe_duration(clip_path)
        if clip_duration <= 0.1:
            raise RuntimeError(f"empty clip generated: {clip_path}")
        if clip_duration > request.seconds + 0.02:
            _ffmpeg("-i", str(clip_path), "-t", f"{request.seconds:.3f}", "-c:v", "copy", str(clips_dir / f"clip_{idx:02d}_trim.mp4"))
            clip_path = clips_dir / f"clip_{idx:02d}_trim.mp4"
        elif clip_duration + 0.02 < request.seconds:
            pad_seconds = max(0.0, request.seconds - clip_duration)
            _ffmpeg(
                "-i",
                str(clip_path),
                "-vf",
                f"tpad=stop_mode=clone:stop_duration={pad_seconds:.3f}",
                "-r",
                str(fps),
                str(clips_dir / f"clip_{idx:02d}_pad.mp4"),
            )
            clip_path = clips_dir / f"clip_{idx:02d}_pad.mp4"
        generated.append(clip_path)

    if status_callback:
        status_callback("stitching")
    concat_list = clips_dir / "concat.txt"
    concat_list.write_text("\n".join([f"file '{p.as_posix()}'" for p in generated]), encoding="utf-8")
    stitched = final_dir / "stitched_raw.mp4"
    _ffmpeg("-f", "concat", "-safe", "0", "-i", str(concat_list), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(stitched))

    target_video = final_dir / "video_out.mp4"
    final_w, final_h = (1920, 1080) if cfg.name in {"balanced", "max"} else (width, height)
    vf_parts = []
    if final_w != width or final_h != height:
        vf_parts.append(f"scale={final_w}:{final_h}:flags=lanczos")
    vf_parts.append(f"fps={fps}")
    if enable_sharpen:
        vf_parts.append("cas=strength=0.25")
    vf = ",".join(vf_parts)
    try:
        _ffmpeg(
            "-i",
            str(stitched),
            "-vf",
            vf,
            "-t",
            f"{total_seconds:.3f}",
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p2" if cfg.name in {"fasttest", "fast"} else "p4",
            str(target_video),
        )
    except Exception:
        vf_fallback = ",".join(part for part in vf_parts if not part.startswith("cas="))
        _ffmpeg(
            "-i",
            str(stitched),
            "-vf",
            vf_fallback,
            "-t",
            f"{total_seconds:.3f}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            str(target_video),
        )

    if status_callback:
        status_callback("muxing")
    silent = final_dir / "silent.wav"
    _ffmpeg("-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000", "-t", f"{total_seconds:.3f}", str(silent))
    final_mp4 = final_dir / "final.mp4"
    _ffmpeg("-i", str(target_video), "-i", str(silent), "-shortest", "-c:v", "copy", "-c:a", "aac", str(final_mp4))

    duration = _probe_duration(final_mp4)
    if math.fabs(duration - total_seconds) > 0.15:
        _ffmpeg("-i", str(final_mp4), "-t", f"{total_seconds:.3f}", "-c:v", "copy", "-c:a", "copy", str(final_dir / "final_trim.mp4"))
        final_mp4 = final_dir / "final_trim.mp4"

    report = {
        "job_id": job_id,
        "backend": provider.name,
        "seconds": total_seconds,
        "fps": fps,
        "clips": len(generated),
        "clip_seconds": clip_seconds,
        "frames_per_clip": frames_per_clip,
        "steps": steps,
        "guidance": guidance,
        "width": width,
        "height": height,
        "preset": cfg.name,
        "post_sharpen": enable_sharpen,
        "elapsed_s": round(time.time() - started, 3),
        "final_video": str(final_mp4),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return final_mp4, report_path
