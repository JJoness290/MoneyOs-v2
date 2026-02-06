from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from app.config import OUTPUT_DIR
from src.moneyos.ai_video.beats import Beat, generate_beats
from src.moneyos.ai_video.finalize import finalize_with_audio
from src.moneyos.ai_video.generator import BackendUnavailable, ClipStaticError, generate_clip
from src.moneyos.ai_video.prompts import beat_to_video_prompt
from src.moneyos.ai_video.stitcher import stitch_clips


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


def run_ai_video_60s(job_id: str, script: str, audio_path: Path) -> AiVideoResult:
    clip_seconds = int(os.getenv("MONEYOS_AI_CLIP_SECONDS", "3"))
    total_seconds = int(os.getenv("MONEYOS_AI_TOTAL_SECONDS", "60"))
    output_dir = _output_dir(job_id)
    clips_dir = output_dir / "clips"
    final_dir = output_dir / "final"
    output_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)
    if not audio_path.exists():
        raise FileNotFoundError(f"audio_path missing: {audio_path}")
    beats = generate_beats(script, total_seconds, clip_seconds)
    if len(beats) != int(total_seconds / clip_seconds):
        raise RuntimeError("Beat generation failed to match expected clip count.")
    clip_paths: list[Path] = []
    backend_name = None
    gpu_used = False
    resolution = None
    fps = None
    static_clips = 0
    for beat in beats:
        prompt = beat_to_video_prompt(beat)
        seed = _seed_for_clip(script, beat.index)
        clip_path = clips_dir / f"clip_{beat.index:02d}.mp4"
        try:
            info = generate_clip(prompt, seed, clip_seconds, clip_path)
        except ClipStaticError:
            static_clips += 1
            raise
        clip_paths.append(clip_path)
        backend_name = info.get("backend")
        resolution = info.get("resolution")
        fps = info.get("fps")
        gpu_used = info.get("device") == "cuda"
    if len(clip_paths) < int(total_seconds / clip_seconds):
        raise RuntimeError("clips_generated < expected count")
    stitched_path = output_dir / "stitched.mp4"
    transition = os.getenv("MONEYOS_AI_TRANSITION", "hard_cut").strip().lower()
    stitch_clips(clip_paths, stitched_path, transition=transition)
    final_path = final_dir / "final.mp4"
    finalize_with_audio(stitched_path, audio_path, final_path)
    if not final_path.exists():
        raise RuntimeError("final.mp4 missing after finalize")
    report_path = final_dir / "report.json"
    report_payload = {
        "mode": "ai_video_only",
        "backend": backend_name or "unknown",
        "gpu_used": bool(gpu_used),
        "clips_generated": len(clip_paths),
        "clip_duration": clip_seconds,
        "total_duration": total_seconds,
        "resolution": resolution or "unknown",
        "fps": fps or 30,
        "audio_path": str(audio_path),
        "output": str(final_path),
        "validation": {
            "motion_verified": static_clips == 0,
            "static_clips": static_clips,
        },
    }
    report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    if report_payload["clips_generated"] < int(total_seconds / clip_seconds):
        raise RuntimeError("clips_generated < expected")
    if not report_payload["validation"]["motion_verified"]:
        raise RuntimeError("motion_verified == false")
    if not final_path.exists():
        raise RuntimeError("output missing")
    return AiVideoResult(output_dir=output_dir, final_video=final_path, report_path=report_path)
