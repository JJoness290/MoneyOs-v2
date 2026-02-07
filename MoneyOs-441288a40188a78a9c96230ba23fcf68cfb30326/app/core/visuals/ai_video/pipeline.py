from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from app.config import OUTPUT_DIR
from app.core.visuals.ai_video.beats import generate_beats
from app.core.visuals.ai_video.finalize import finalize_with_audio
from app.core.visuals.ai_video.generator import BackendUnavailable, ClipStaticError, generate_clip
from app.core.visuals.ai_video.prompts import beat_to_video_prompt
from app.core.visuals.ai_video.reporting import write_report
from app.core.visuals.ai_video.stitcher import stitch_clips


StatusCallback = Callable[[str], None] | None


@dataclass(frozen=True)
class AiVideoResult:
    output_dir: Path
    final_video: Path
    report_path: Path


def _seed_for_clip(script: str, index: int) -> int:
    digest = hashlib.sha256(f"{script}:{index}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _output_dir(job_id: str) -> Path:
    return OUTPUT_DIR / "ai_video" / job_id


def _gpu_name() -> str | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        return None
    return None


def run_ai_video_job(
    job_id: str,
    script: str,
    audio_path: Path,
    status_callback: StatusCallback = None,
) -> AiVideoResult:
    clip_seconds = int(os.getenv("MONEYOS_AI_CLIP_SECONDS", "3"))
    total_seconds = int(os.getenv("MONEYOS_AI_TOTAL_SECONDS", "60"))
    fps = int(os.getenv("MONEYOS_AI_FPS", "16"))
    width = int(os.getenv("MONEYOS_AI_WIDTH", "1024"))
    height = int(os.getenv("MONEYOS_AI_HEIGHT", "576"))
    steps = int(os.getenv("MONEYOS_AI_STEPS", "30"))
    guidance = float(os.getenv("MONEYOS_AI_GUIDANCE", "6.0"))
    output_dir = _output_dir(job_id)
    clips_dir = output_dir / "clips"
    final_dir = output_dir / "final"
    report_path = final_dir / "report.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)
    if not audio_path.exists():
        raise FileNotFoundError(f"audio_path missing: {audio_path}")
    beats = generate_beats(script, total_seconds, clip_seconds)
    expected = int(total_seconds / clip_seconds)
    if len(beats) != expected:
        raise RuntimeError("Beat generation failed to match expected clip count.")
    clip_paths: list[Path] = []
    backend_name = None
    gpu_used = False
    resolution = None
    static_clips = 0
    regenerations = 0
    start_time = time.time()
    try:
        if status_callback:
            status_callback("Generating AI video clips")
        for beat in beats:
            prompt_pack = beat_to_video_prompt(beat)
            seed = _seed_for_clip(script, beat.index)
            clip_path = clips_dir / f"clip_{beat.index:02d}.mp4"
            try:
                info = generate_clip(
                    prompt_pack=prompt_pack,
                    seed=seed,
                    seconds=clip_seconds,
                    out_path=clip_path,
                    fps=fps,
                    width=width,
                    height=height,
                    steps=steps,
                    guidance=guidance,
                )
            except ClipStaticError:
                static_clips += 1
                raise
            clip_paths.append(clip_path)
            backend_name = info.backend
            resolution = info.resolution
            gpu_used = info.device == "cuda"
            regenerations += info.regenerations
            if info.duration_s < clip_seconds - 0.25 or info.duration_s > clip_seconds + 0.25:
                raise RuntimeError(f"Clip duration out of bounds: {clip_path} ({info.duration_s:.2f}s)")
        if len(clip_paths) < expected:
            raise RuntimeError("clips_generated < expected count")
        if status_callback:
            status_callback("Stitching AI clips")
        stitched_path = output_dir / "stitch" / "video_noaudio.mp4"
        transition = os.getenv("MONEYOS_AI_TRANSITION", "crossfade").strip().lower()
        stitch_clips(clip_paths, stitched_path, transition=transition)
        if status_callback:
            status_callback("Finalizing AI video")
        final_path = final_dir / "final.mp4"
        finalize_with_audio(stitched_path, audio_path, final_path)
        if not final_path.exists():
            raise RuntimeError("final.mp4 missing after finalize")
        elapsed = time.time() - start_time
        payload = {
            "mode": "ai_video_only",
            "backend": backend_name or "unknown",
            "gpu_used": bool(gpu_used),
            "gpu_name": _gpu_name(),
            "clips_expected": expected,
            "clips_generated": len(clip_paths),
            "clip_duration_s": clip_seconds,
            "total_target_s": total_seconds,
            "fps": fps,
            "resolution": resolution or f"{width}x{height}",
            "steps": steps,
            "guidance": guidance,
            "audio_path": str(audio_path),
            "outputs": {
                "final_mp4": str(final_path),
                "video_noaudio": str(stitched_path),
                "clips_dir": str(clips_dir),
            },
            "validation": {
                "motion_verified": static_clips == 0,
                "static_clips": static_clips,
                "regenerations": regenerations,
            },
            "timings": {
                "total_s": elapsed,
                "per_clip_avg_s": elapsed / max(len(clip_paths), 1),
            },
        }
        write_report(report_path, payload)
        if payload["clips_generated"] < expected:
            raise RuntimeError("clips_generated < expected")
        if not payload["validation"]["motion_verified"]:
            raise RuntimeError("motion_verified == false")
        if not final_path.exists():
            raise RuntimeError("output missing")
        return AiVideoResult(output_dir=output_dir, final_video=final_path, report_path=report_path)
    except Exception as exc:  # noqa: BLE001
        elapsed = time.time() - start_time
        error_payload = {
            "mode": "ai_video_only",
            "backend": backend_name or os.getenv("MONEYOS_AI_VIDEO_BACKEND", "unknown"),
            "gpu_used": bool(gpu_used),
            "gpu_name": _gpu_name(),
            "clips_expected": expected,
            "clips_generated": len(clip_paths),
            "clip_duration_s": clip_seconds,
            "total_target_s": total_seconds,
            "fps": fps,
            "resolution": resolution or f"{width}x{height}",
            "steps": steps,
            "guidance": guidance,
            "audio_path": str(audio_path),
            "outputs": {
                "clips_dir": str(clips_dir),
                "final_dir": str(final_dir),
            },
            "errors": [str(exc)],
            "timings": {
                "total_s": elapsed,
            },
        }
        write_report(report_path, error_payload)
        raise
