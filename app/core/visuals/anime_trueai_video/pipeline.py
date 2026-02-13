from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import time

from app.config import OUTPUT_DIR
from app.core.visuals.anime_trueai_video.cogvideox_provider import CogVideoXProvider
from app.core.visuals.anime_trueai_video.provider import ClipRequest, TextToVideoProvider

StatusCallback = callable


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


def run_trueai_60s_job(job_id: str, prompt: str, status_callback=None) -> tuple[Path, Path]:
    clip_count = 10
    clip_seconds = 6
    total_seconds = 60.0
    fps = 24
    width = 1280
    height = 720
    steps = int(os.getenv("MONEYOS_TRUEAI_STEPS", "30"))
    guidance = float(os.getenv("MONEYOS_TRUEAI_GUIDANCE", "6.0"))
    seed = int(os.getenv("MONEYOS_TRUEAI_SEED", "777"))
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

    generated: list[Path] = []
    started = time.time()
    for idx in range(clip_count):
        if status_callback:
            status_callback(f"plan → generating clip {idx + 1}/{clip_count}")
        clip_path = clips_dir / f"clip_{idx:02d}.mp4"
        request = ClipRequest(
            prompt=_anime_prompt(prompt, character_desc, idx),
            negative_prompt=_negative_prompt(),
            seed=seed,
            seconds=clip_seconds,
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
                width=960,
                height=540,
                steps=max(20, request.steps - 8),
                guidance=request.guidance,
                out_path=low_clip,
            )
            provider.generate(request)
            _ffmpeg("-i", str(low_clip), "-vf", "scale=1280:720", "-r", str(fps), str(clip_path))
        if _probe_duration(clip_path) <= 0.1:
            raise RuntimeError(f"empty clip generated: {clip_path}")
        generated.append(clip_path)

    if status_callback:
        status_callback("stitching")
    concat_list = clips_dir / "concat.txt"
    concat_list.write_text("\n".join([f"file '{p.as_posix()}'" for p in generated]), encoding="utf-8")
    stitched = final_dir / "stitched_720.mp4"
    _ffmpeg("-f", "concat", "-safe", "0", "-i", str(concat_list), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(stitched))

    target_video = final_dir / "video_1080.mp4"
    _ffmpeg("-i", str(stitched), "-vf", "scale=1920:1080,fps=24", "-t", f"{total_seconds:.3f}", "-c:v", "h264_nvenc", "-preset", "p4", str(target_video))

    if status_callback:
        status_callback("muxing")
    silent = final_dir / "silence.wav"
    _ffmpeg("-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000", "-t", f"{total_seconds:.3f}", str(silent))
    final_mp4 = final_dir / "final.mp4"
    _ffmpeg("-i", str(target_video), "-i", str(silent), "-shortest", "-c:v", "copy", "-c:a", "aac", str(final_mp4))

    duration = _probe_duration(final_mp4)
    if math.fabs(duration - total_seconds) > 0.08:
        _ffmpeg("-i", str(final_mp4), "-t", f"{total_seconds:.3f}", "-c:v", "copy", "-c:a", "copy", str(final_dir / "final_fixed.mp4"))
        final_mp4 = final_dir / "final_fixed.mp4"

    report = {
        "ok": True,
        "backend": provider.name,
        "clips": len(generated),
        "fps": fps,
        "target_seconds": total_seconds,
        "final_video": str(final_mp4),
        "character_description": character_desc,
        "elapsed_s": time.time() - started,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return final_mp4, report_path
