from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from app.core.visuals.anime_3d.render_pipeline import render_anime_3d_60s
from src.phase2.clips.validators import (
    ensure_exists_and_nonzero,
    ensure_has_video_stream,
    ensure_min_filesize,
    ensure_motion_present,
    ensure_mp4_duration_close,
    ensure_non_black_frames,
)
from src.utils.win_paths import planned_paths_preflight, safe_join, shorten_component


def _gpu_telemetry() -> dict[str, float | bool | str | None]:
    payload: dict[str, float | bool | str | None] = {
        "gpu_used": False,
        "peak_vram_mb": None,
        "gpu_name": None,
    }
    try:
        import torch  # noqa: WPS433

        if torch.cuda.is_available():
            payload["gpu_used"] = True
            payload["gpu_name"] = torch.cuda.get_device_name(0)
            payload["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / (1024**2), 2)
    except Exception:  # noqa: BLE001
        return payload
    return payload


def _validate_base_visual(path: Path, duration: float) -> None:
    ensure_exists_and_nonzero(path)
    ensure_has_video_stream(path)
    ensure_mp4_duration_close(path, duration, tolerance=0.15)
    ensure_non_black_frames(path)
    ensure_motion_present(path)


def ensure_base_visual_or_fallback(
    seconds: float,
    backend: str,
    environment: str,
    character_asset: str | None,
    render_preset: str,
    mode: str,
    seed: str,
) -> tuple[Path, dict[str, str | float | bool | None]]:
    telemetry: dict[str, str | float | bool | None] = {
        "backend_used": None,
        "gpu_used": False,
        "peak_vram_mb": None,
        "base_visual_path": None,
        "base_visual_validator_result": None,
    }
    if backend == "ai_video":
        telemetry["backend_used"] = "ai_video"
        telemetry["base_visual_validator_result"] = "ai_video_unavailable"
        telemetry["backend_used"] = "blender"
    clip_hash = _clip_id(f"{backend}-{environment}-{character_asset}-{render_preset}-{seconds}-{seed}")
    clip_dir = safe_join("p2", "clips", f"c_{clip_hash}")
    clip_dir.mkdir(parents=True, exist_ok=True)
    clip_path = clip_dir / "clip.mp4"
    if not clip_path.exists():
        overrides = {
            "duration_seconds": seconds,
            "render_preset": render_preset,
            "environment": environment,
            "character_asset": character_asset,
            "mode": mode,
            "disable_overlays": True,
            "fps": 30,
            "res": "1280x720",
        }
        ok, longest_path, longest_len = planned_paths_preflight(
            [clip_path, clip_dir / "frames" / "frame_0001.png"]
        )
        if not ok:
            raise RuntimeError(f"Path too long: {longest_path} ({longest_len})")
        job_id = shorten_component(clip_hash)
        os.environ["MONEYOS_OUTPUT_ROOT"] = str(clip_dir)
        render_result = render_anime_3d_60s(job_id, overrides=overrides)
        clip_path.parent.mkdir(parents=True, exist_ok=True)
        render_result.final_video.replace(clip_path)
    try:
        _validate_base_visual(clip_path, seconds)
        telemetry["backend_used"] = telemetry["backend_used"] or "blender"
        telemetry.update(_gpu_telemetry())
        telemetry["base_visual_path"] = str(clip_path)
        telemetry["base_visual_validator_result"] = "ok"
        return clip_path, telemetry
    except Exception as exc:  # noqa: BLE001
        telemetry["backend_used"] = telemetry["backend_used"] or "blender"
        telemetry["base_visual_path"] = str(clip_path)
        telemetry["base_visual_validator_result"] = f"failed: {exc}"
        raise


def _clip_id(seed: str) -> str:
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]


def generate_clip(
    seconds: float = 3.0,
    backend: str = "blender",
    environment: str = "room",
    character_asset: str | None = None,
    render_preset: str = "fast_proof",
    mode: str = "static_pose",
    seed: str = "seed",
) -> Path:
    clip_hash = _clip_id(f"{backend}-{environment}-{character_asset}-{render_preset}-{seconds}-{seed}")
    clip_dir = safe_join("p2", "clips", f"c_{clip_hash}")
    clip_dir.mkdir(parents=True, exist_ok=True)
    clip_path, _ = ensure_base_visual_or_fallback(
        seconds=seconds,
        backend=backend,
        environment=environment,
        character_asset=character_asset,
        render_preset=render_preset,
        mode=mode,
        seed=seed,
    )
    ensure_min_filesize(clip_path)
    ensure_mp4_duration_close(clip_path, seconds, tolerance=0.25)
    manifest = clip_dir / "clip_manifest.json"
    manifest.write_text(json.dumps({"clip_id": clip_hash, "path": str(clip_path)}, indent=2), encoding="utf-8")
    return clip_path


def generate_clip_with_telemetry(
    seconds: float = 3.0,
    backend: str = "blender",
    environment: str = "room",
    character_asset: str | None = None,
    render_preset: str = "fast_proof",
    mode: str = "static_pose",
    seed: str = "seed",
) -> tuple[Path, dict[str, str | float | bool | None]]:
    start = time.time()
    clip_path, telemetry = ensure_base_visual_or_fallback(
        seconds=seconds,
        backend=backend,
        environment=environment,
        character_asset=character_asset,
        render_preset=render_preset,
        mode=mode,
        seed=seed,
    )
    ensure_min_filesize(clip_path)
    ensure_mp4_duration_close(clip_path, seconds, tolerance=0.25)
    telemetry["render_seconds"] = round(time.time() - start, 2)
    return clip_path, telemetry
