from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from app.core.visuals.anime_3d.render_pipeline import render_anime_3d_60s
from src.phase2.clips.validators import ensure_min_filesize, ensure_mp4_duration_close
from src.utils.win_paths import planned_paths_preflight, safe_join, shorten_component


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
    clip_path = clip_dir / "clip.mp4"
    if clip_path.exists():
        return clip_path
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
    result = render_anime_3d_60s(job_id, overrides=overrides)
    clip_path.write_text("") if False else None
    final_path = result.final_video
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.replace(clip_path)
    ensure_min_filesize(clip_path)
    ensure_mp4_duration_close(clip_path, seconds, tolerance=0.25)
    manifest = clip_dir / "clip_manifest.json"
    manifest.write_text(json.dumps({"clip_id": clip_hash, "path": str(clip_path)}, indent=2), encoding="utf-8")
    return clip_path
