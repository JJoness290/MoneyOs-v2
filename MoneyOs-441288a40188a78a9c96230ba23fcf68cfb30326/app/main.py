from __future__ import annotations

import json
import os
import threading
import asyncio
import uuid
import hashlib
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from app.config import (
    ANIME3D_ASSET_MODE,
    ANIME3D_POSTFX,
    ANIME3D_QUALITY,
    ANIME3D_STYLE_PRESET,
    ANIME3D_TEXTURE_MODE,
    ANIME3D_OUTLINE_MODE,
    ANIME3D_RESOLUTION,
    AUTO_CHARACTERS_DIR,
    CHARACTERS_DIR,
    OUTPUT_DIR,
    SD_MODEL_PATH,
    VIDEO_DIR,
    VISUAL_MODE,
)
from app.core.paths import get_assets_root, get_output_root, get_repo_root
from app.core.assets.harvester.cache import get_cache_paths
from app.core.assets.harvester.harvester import harvest_assets
from app.core.autopilot import enqueue as autopilot_enqueue, start_autopilot, status as autopilot_status
from app.core.bootstrap import ensure_dependencies
from app.core.anime_episode import EpisodeResult, generate_anime_episode_10m
from app.core.visuals.anime_3d.animation_library import rebuild_animation_library
from app.core.visuals.anime_3d.blender_runner import detect_blender
from app.core.visuals.anime_3d.render_pipeline import (
    Anime3DResult,
    anime_3d_output_dir,
    finalize_anime_3d,
    render_anime_3d_60s,
)
from app.core.pipeline import PipelineResult, run_pipeline
from app.core.system_specs import get_system_specs
from src.utils.phase import is_phase2_or_higher, normalize_phase

app = FastAPI()

STATUS_IDLE = "Idle"
STATUS_SCRIPT = "Generating script..."
STATUS_BROLL = "Downloading B-roll..."
STATUS_RENDER = "Rendering video..."
STATUS_DONE = "Done"

_jobs_lock = threading.Lock()
_jobs: Dict[str, dict] = {}
_last_clip_telemetry: dict[str, object] = {}
_last_job_snapshot: dict[str, object] = {}
_perf_lock = threading.Lock()
_perf_history_path = OUTPUT_DIR / "perf_history.json"

_STAGE_ORDER = ["script", "broll", "render", "audio", "blender", "frames", "encode", "mux", "done"]
_STAGE_ALIASES = {
    "generating script": "script",
    "downloading b-roll": "broll",
    "rendering video": "render",
    "generating anime episode": "script",
    "queued 3d render": "script",
    "rendering anime 3d episode": "render",
    "generating audio": "audio",
    "audio": "audio",
    "blender": "blender",
    "rendering frames": "frames",
    "encoding video": "encode",
    "muxing audio": "mux",
    "queued": "script",
    "done": "done",
    "complete": "done",
}
_STAGE_PROGRESS = {
    "script": 10,
    "broll": 40,
    "render": 85,
    "audio": 10,
    "blender": 20,
    "frames": 50,
    "encode": 95,
    "mux": 98,
    "done": 100,
}


class AnimeEpisodeRequest(BaseModel):
    topic_hint: Optional[str] = None
    lane: Optional[str] = None


class Anime3DRequest(BaseModel):
    duration_seconds: Optional[float] = None
    duration_s: Optional[float] = None
    fps: Optional[int] = None
    res: Optional[str] = None
    quality: Optional[str] = None
    style_preset: Optional[str] = None
    outline_mode: Optional[str] = None
    postfx: Optional[bool] = None
    vfx_emission_strength: Optional[float] = None
    vfx_scale: Optional[float] = None
    vfx_screen_coverage: Optional[float] = None
    render_preset: Optional[str] = None
    environment: Optional[str] = None
    character_asset: Optional[str] = None
    disable_overlays: Optional[bool] = None
    mode: Optional[str] = None
    strict_assets: Optional[bool] = None


@app.on_event("startup")
def bootstrap_dependencies() -> None:
    ensure_dependencies()
    start_autopilot()
    try:
        from app.core.visuals.ffmpeg_utils import select_video_encoder  # noqa: WPS433
        from app.config import performance  # noqa: WPS433

        cuda_available = False
        try:
            import torch  # noqa: WPS433

            cuda_available = torch.cuda.is_available()
        except Exception:  # noqa: BLE001
            cuda_available = False
        encoder_args, encoder = select_video_encoder()
        _ = encoder_args
        print(
            "[GPU] "
            f"cuda_available={cuda_available} "
            f"MONEYOS_USE_GPU={os.getenv('MONEYOS_USE_GPU', 'auto')} "
            f"encoder_selected={encoder} "
            f"ffmpeg_threads={performance.ffmpeg_threads()} "
            f"ram_mode={performance.ram_mode()}"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[GPU] status unavailable: {exc}")


def _format_mmss(seconds: float) -> str:
    total_seconds = int(round(seconds))
    minutes, secs = divmod(total_seconds, 60)
    return f"{minutes:02d}:{secs:02d}"


def compute_clip_id(cache_payload: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    canonical = json.dumps(cache_payload, sort_keys=True, separators=(",", ":"))
    cache_hash = hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]
    clip_id = f"c_{cache_hash}"
    return clip_id, cache_payload


def md5_file(path: Union[str, os.PathLike], chunk_size: int = 1024 * 1024) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"md5_file: missing file: {path}")
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_dir(path: Union[str, os.PathLike]) -> None:
    os.makedirs(path, exist_ok=True)


def _load_perf_history() -> dict:
    if not _perf_history_path.exists():
        return {}
    try:
        return json.loads(_perf_history_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save_perf_history(payload: dict) -> None:
    _perf_history_path.parent.mkdir(parents=True, exist_ok=True)
    _perf_history_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _stage_key(status: str) -> str:
    lowered = status.lower()
    for phrase, key in _STAGE_ALIASES.items():
        if phrase in lowered:
            return key
    return "script"


def _record_stage_duration(stage: str, duration_s: float) -> None:
    with _perf_lock:
        history = _load_perf_history()
        stage_entry = history.get(stage, {"avg_seconds": 0.0, "count": 0})
        count = stage_entry["count"] + 1
        avg = (stage_entry["avg_seconds"] * stage_entry["count"] + duration_s) / count
        history[stage] = {"avg_seconds": avg, "count": count}
        _save_perf_history(history)


def _estimate_eta(stage: str) -> float | None:
    history = _load_perf_history()
    if stage not in history:
        return None
    if stage not in _STAGE_ORDER:
        return None
    remaining = 0.0
    for stage_key in _STAGE_ORDER[_STAGE_ORDER.index(stage) :]:
        remaining += history.get(stage_key, {}).get("avg_seconds", 0.0)
    return remaining if remaining > 0 else None


def _set_status(
    job_id: str,
    status: str,
    result: Optional[PipelineResult] = None,
    episode_result: Optional[EpisodeResult] = None,
    anime_3d_result: Optional[Anime3DResult] = None,
    stage_key: str | None = None,
    progress_pct: int | None = None,
    extra: dict | None = None,
) -> None:
    with _jobs_lock:
        payload = _jobs.setdefault(job_id, {"status": STATUS_IDLE})
        resolved_stage = stage_key or _stage_key(status)
        now = datetime.now()
        previous_stage = payload.get("stage_key")
        previous_start = payload.get("stage_started_at")
        if previous_stage and previous_start and previous_stage != resolved_stage:
            elapsed = now.timestamp() - previous_start
            _record_stage_duration(previous_stage, elapsed)
            payload["stage_started_at"] = now.timestamp()
        elif previous_stage is None:
            payload["stage_started_at"] = now.timestamp()
        payload["stage_key"] = resolved_stage
        payload["status"] = status
        payload["stage"] = status
        payload["updated_at"] = now.isoformat()
        payload["progress_pct"] = progress_pct if progress_pct is not None else _STAGE_PROGRESS.get(resolved_stage, 5)
        eta_seconds = _estimate_eta(resolved_stage)
        payload["eta_seconds"] = eta_seconds
        payload["eta_mmss"] = _format_mmss(eta_seconds) if eta_seconds is not None else None
        if extra:
            payload.update(extra)
        if result:
            payload["video_path"] = str(result.video.output_path.resolve())
            payload["duration"] = result.video.duration_seconds
            payload["duration_mmss"] = _format_mmss(result.video.duration_seconds)
            payload["audio_duration"] = result.tts.duration_seconds
            payload["audio_duration_mmss"] = _format_mmss(result.tts.duration_seconds)
            payload["script_path"] = str(result.script_path.resolve())
            payload["word_count"] = result.word_count
            payload["titles"] = result.titles
            payload["description"] = result.description
            payload["success"] = True
        if episode_result:
            payload["video_path"] = str(episode_result.video_path.resolve())
            payload["duration"] = episode_result.total_video_seconds
            payload["duration_mmss"] = _format_mmss(episode_result.total_video_seconds)
            payload["audio_duration"] = episode_result.total_audio_seconds
            payload["audio_duration_mmss"] = _format_mmss(episode_result.total_audio_seconds)
            payload["output_dir"] = str(episode_result.output_dir.resolve())
            payload["success"] = True
        if anime_3d_result:
            payload["video_path"] = str(anime_3d_result.final_video.resolve())
            payload["final_video"] = str(anime_3d_result.final_video.resolve())
            payload["output_dir"] = str(anime_3d_result.output_dir.resolve())
            payload["audio_path"] = str(anime_3d_result.audio_path.resolve())
            payload["duration"] = anime_3d_result.duration_seconds
            payload["duration_mmss"] = _format_mmss(anime_3d_result.duration_seconds)
            payload["warnings"] = anime_3d_result.warnings
            payload["success"] = True


def _tail_file(path: Path, max_chars: int = 2000) -> str | None:
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")[-max_chars:]
    except Exception:  # noqa: BLE001
        return None


def _set_error(job_id: str, message: str, extra: dict | None = None) -> None:
    with _jobs_lock:
        payload = _jobs.setdefault(job_id, {"status": STATUS_IDLE})
        payload["status"] = message
        payload["progress_pct"] = payload.get("progress_pct", 0)
        payload["success"] = False
        payload["updated_at"] = datetime.now().isoformat()
        payload["stage_key"] = "error"
        payload["stage"] = "Error"
        if extra:
            payload.update(extra)


def _run_job(job_id: str) -> None:
    try:
        result = run_pipeline(lambda status: _set_status(job_id, status))
        _set_status(job_id, STATUS_DONE, result)
    except Exception as exc:  # noqa: BLE001
        _set_error(job_id, f"Error: {exc}")


def _run_anime_episode(job_id: str, topic_hint: str | None, lane: str | None) -> None:
    try:
        result = generate_anime_episode_10m(
            lambda status: _set_status(job_id, status),
            topic_hint=topic_hint,
            lane=lane,
        )
        _set_status(job_id, STATUS_DONE, episode_result=result)
    except Exception as exc:  # noqa: BLE001
        _set_error(job_id, f"Error: {exc}")


def _run_anime_3d_60s(job_id: str, req: Anime3DRequest) -> None:
    def _update(payload: dict) -> None:
        _set_status(
            job_id,
            payload.get("status", "Rendering anime 3D episode..."),
            stage_key=payload.get("stage_key"),
            progress_pct=payload.get("progress_pct"),
            extra=payload.get("extra"),
        )

    try:
        _set_status(job_id, "Generating audio", stage_key="audio", progress_pct=5)
        overrides = req.dict(exclude_none=True)
        if "duration_seconds" not in overrides and "duration_s" in overrides:
            overrides["duration_seconds"] = overrides["duration_s"]
        result = render_anime_3d_60s(
            job_id,
            status_callback=_update,
            overrides=overrides,
        )
        phase = normalize_phase(os.getenv("MONEYOS_PHASE"))
        strict_vfx = os.getenv("MONEYOS_STRICT_VFX", "0") == "1"
        if "vfx_emissive_check_failed" in result.warnings and not strict_vfx and not is_phase2_or_higher(phase):
            status_text = "Complete (VFX skipped)"
        else:
            status_text = "Complete"
        _set_status(job_id, status_text, anime_3d_result=result, stage_key="done", progress_pct=100)
    except Exception as exc:  # noqa: BLE001
        output_dir = anime_3d_output_dir(job_id)
        extra = {
            "output_dir": str(output_dir.resolve()),
            "blender_cmd": str((output_dir / "blender_cmd.txt").resolve()),
            "blender_stdout": str((output_dir / "blender_stdout.txt").resolve()),
            "blender_stderr": str((output_dir / "blender_stderr.txt").resolve()),
            "blender_stderr_tail": _tail_file(output_dir / "blender_stderr.txt"),
            "blender_stdout_tail": _tail_file(output_dir / "blender_stdout.txt"),
        }
        _set_error(job_id, f"Error: {exc}", extra=extra)


def _run_anime_3d_clip(job_id: str, req: Anime3DRequest) -> None:
    from src.phase2.clips.clip_generator import generate_clip_with_telemetry  # noqa: WPS433

    try:
        clip_path, telemetry = generate_clip_with_telemetry(
            seconds=req.duration_seconds or 3.0,
            environment=req.environment or "room",
            character_asset=req.character_asset,
            render_preset=req.render_preset or "fast_proof",
            mode=req.mode or "static_pose",
            seed=job_id,
        )
        with _jobs_lock:
            _last_clip_telemetry.update(telemetry)
        _set_status(
            job_id,
            "Complete",
            stage_key="done",
            progress_pct=100,
            extra={
                "clip": str(clip_path),
                "backend_used_last_clip": telemetry.get("backend_used"),
                "gpu_used_last_clip": telemetry.get("gpu_used"),
                "peak_vram_mb_last_clip": telemetry.get("peak_vram_mb"),
                "base_visual_path_last_clip": telemetry.get("base_visual_path"),
                "base_visual_validator_result": telemetry.get("base_visual_validator_result"),
            },
        )
    except Exception as exc:  # noqa: BLE001
        _set_error(job_id, f"Error: {exc}")


@app.get("/")
async def ui() -> HTMLResponse:
    index_path = Path(__file__).parent / "ui" / "index.html"
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.get("/api/system/specs")
async def system_specs() -> JSONResponse:
    return JSONResponse(get_system_specs())


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.get("/debug/status")
async def debug_status() -> JSONResponse:
    blender = detect_blender()
    try:
        from src.utils.capabilities import capabilities_snapshot  # noqa: WPS433
        from src.utils.win_paths import get_short_workdir  # noqa: WPS433

        caps = capabilities_snapshot(get_short_workdir() / "p2" / "cache" / "capabilities.json")
    except Exception as exc:  # noqa: BLE001
        caps = {"error": str(exc)}
    assets_root = get_assets_root()
    required_assets = {
        "characters/hero.blend": (assets_root / "characters" / "hero.blend"),
        "characters/enemy.blend": (assets_root / "characters" / "enemy.blend"),
        "envs/city.blend": (assets_root / "envs" / "city.blend"),
        "anims/idle.fbx": (assets_root / "anims" / "idle.fbx"),
        "anims/run.fbx": (assets_root / "anims" / "run.fbx"),
        "anims/punch.fbx": (assets_root / "anims" / "punch.fbx"),
        "vfx/explosion.png": (assets_root / "vfx" / "explosion.png"),
        "vfx/energy_arc.png": (assets_root / "vfx" / "energy_arc.png"),
        "vfx/smoke.png": (assets_root / "vfx" / "smoke.png"),
    }
    vram_gb = None
    payload = {
        "autopilot": autopilot_status(),
        "visual_mode": VISUAL_MODE,
        "visual_backend": os.getenv("MONEYOS_VISUAL_BACKEND", "hybrid"),
        "cwd": str(Path.cwd()),
        "repo_root": str(get_repo_root()),
        "assets_root": str(assets_root),
        "output_root": str(get_output_root()),
        "assets_ready": {key: path.exists() for key, path in required_assets.items()},
        "assets_missing": [key for key, path in required_assets.items() if not path.exists()],
        "asset_mode": ANIME3D_ASSET_MODE,
        "texture_mode": ANIME3D_TEXTURE_MODE,
        "sd_model_used": SD_MODEL_PATH,
        "texture_resolution": f"{ANIME3D_RESOLUTION[0]}x{ANIME3D_RESOLUTION[1]}",
        "style_preset": ANIME3D_STYLE_PRESET,
        "outline_mode": ANIME3D_OUTLINE_MODE,
        "postfx": ANIME3D_POSTFX,
        "quality": ANIME3D_QUALITY,
        "blender": {
            "found": blender.found,
            "path": blender.path,
            "version": blender.version,
            "error": blender.error,
        },
        "capabilities": caps,
        "anime_3d_ready": blender.found and VISUAL_MODE == "anime_3d",
        "last_error": blender.error,
        "last_clip_telemetry": _last_clip_telemetry,
        "last_job_snapshot": _last_job_snapshot,
    }
    try:
        import torch  # noqa: WPS433

        if torch.cuda.is_available():
            vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        payload["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
        payload["vram_detected"] = vram_gb
    except Exception as exc:  # noqa: BLE001
        payload["torch"] = {"error": str(exc)}
    return JSONResponse(payload)


def _resolve_phase_target_seconds() -> tuple[str, float, str | None]:
    raw_phase = os.getenv("MONEYOS_PHASE")
    phase_env = normalize_phase(raw_phase)
    target_env = os.getenv("MONEYOS_TARGET_SECONDS")
    default_seconds = 30.0 if phase_env in {"phase25", "production"} else 600.0
    if target_env:
        try:
            target_value = float(target_env)
        except ValueError:
            target_value = default_seconds
    else:
        target_value = default_seconds
    return phase_env, target_value, raw_phase


def _build_phase25_shot_plan(target_seconds: float) -> list[dict]:
    shot_count = max(6, int(round(target_seconds / 5.0)))
    base = target_seconds / shot_count
    durations = [base for _ in range(shot_count)]
    total = sum(durations)
    if durations:
        durations[-1] += target_seconds - total
    environments = ["room", "street", "studio", "room", "street", "studio"]
    modes = ["wide", "medium", "close", "wide", "medium", "close"]
    plan = []
    for idx, duration in enumerate(durations, start=1):
        plan.append(
            {
                "duration": duration,
                "environment": environments[idx % len(environments)],
                "shot": modes[idx % len(modes)],
                "seed_tag": f"shot-{idx}",
            }
        )
    return plan


def _run_hybrid_episode(job_id: str, target_seconds: float | None = None) -> None:
    from src.phase2.clips.clip_generator import generate_clip_with_telemetry  # noqa: WPS433
    from src.phase2.episodes.episode_assembler import (
        assemble_episode,
        create_silent_audio,
        enforce_duration_contract,
        get_video_duration_seconds,
    )
    from src.phase2.episodes.episode_planner import plan_episode  # noqa: WPS433
    from src.utils.win_paths import safe_join  # noqa: WPS433
    from app.core.visuals.ffmpeg_utils import run_ffmpeg  # noqa: WPS433

    try:
        phase_env, resolved_target, raw_phase = _resolve_phase_target_seconds()
        target_seconds = target_seconds or resolved_target
        clips: list[Path] = []
        accepted: list[dict[str, object]] = []
        seen_md5: set[str] = set()
        diversity_threshold = 0.35
        max_attempts_per_shot = 8
        base_seed_env = os.getenv("MONEYOS_BASE_SEED")
        if base_seed_env and base_seed_env.strip().isdigit():
            base_seed = int(base_seed_env.strip())
        else:
            base_seed = int(time.time())
        output_root = get_output_root()
        phase25_root = output_root / "p2"
        clips_root = phase25_root / "clips"
        tmp_root = phase25_root / "tmp"
        ensure_dir(clips_root)
        ensure_dir(tmp_root)
        if phase_env in {"phase25", "production"}:
            plan = _build_phase25_shot_plan(target_seconds)
            from src.phase2.clips.ai_similarity_validator import (  # noqa: WPS433
                compare_against_accepted,
                embed_frames_with_clip,
                extract_frames_ffmpeg,
                format_similarity_log,
            )
            from src.phase2.clips.diversity_scorer import score_diversity  # noqa: WPS433
            from src.phase2.clips.similarity_memory import SimilarityMemory  # noqa: WPS433
            from src.phase2.director.ai_director import AiDirector  # noqa: WPS433

            similarity_memory = SimilarityMemory(output_root / "p2" / "similarity_memory.json")
            similarity_memory.load()
            director = AiDirector()
            recent_meta: list[dict[str, str]] = []
            accepted_embeddings = similarity_memory.get_accepted_embeddings()
        else:
            plan = plan_episode(target_seconds)
        clips_expected = len(plan)
        print(
            "[PHASE] "
            f"phase={phase_env} "
            f"raw_phase={raw_phase or 'none'} "
            f"target_seconds={target_seconds:.2f}"
        )
        print(
            "[SHOT_PLAN] "
            f"shots={clips_expected} "
            f"shot_seconds={[round(shot['duration'], 2) for shot in plan]} "
            f"total={sum(shot['duration'] for shot in plan):.2f}"
        )
        _set_status(
            job_id,
            "Generating hybrid episode clips...",
            stage_key="visuals",
            progress_pct=5,
            extra={
                "job_type": "anime-episode-10m-hybrid",
                "target_seconds": target_seconds,
                "clips_expected": clips_expected,
                "clips_done": 0,
                "visual_backend": os.getenv("MONEYOS_VISUAL_BACKEND", "hybrid"),
            },
        )
        for index, beat in enumerate(plan, start=1):
            if phase_env not in {"phase25", "production"}:
                clip_path, telemetry = generate_clip_with_telemetry(
                    seconds=float(beat["duration"]),
                    backend=os.getenv("MONEYOS_VISUAL_BACKEND", "hybrid"),
                    environment=beat["environment"],
                    mode=beat["shot"],
                    seed=f"{job_id}-{beat.get('seed_tag', index)}",
                )
                clips.append(clip_path)
                with _jobs_lock:
                    _last_clip_telemetry.update(telemetry)
                _set_status(
                    job_id,
                    f"Rendered clip {index}/{clips_expected}",
                    stage_key="visuals",
                    progress_pct=min(95, int((index / max(1, clips_expected)) * 90)),
                    extra={
                        "clips_done": index,
                        "backend_used_last_clip": telemetry.get("backend_used"),
                        "gpu_used_last_clip": telemetry.get("gpu_used"),
                        "peak_vram_mb_last_clip": telemetry.get("peak_vram_mb"),
                        "base_visual_path_last_clip": telemetry.get("base_visual_path"),
                        "base_visual_validator_result": telemetry.get("base_visual_validator_result"),
                    },
                )
                continue
            attempts = 0
            environment = beat["environment"]
            mode = beat["shot"]
            duration_s = float(beat["duration"])
            seed_value = base_seed + index * 10007
            last_plan: dict[str, str] | None = None
            drop_used = False
            while attempts < max_attempts_per_shot:
                shot_plan = director.next_shot_plan(index, recent_meta)
                if attempts > 0 and last_plan is not None:
                    force_extreme = attempts >= max_attempts_per_shot - 2
                    shot_plan = director.mutate_plan(
                        last_plan,
                        min_mutations=2,
                        force_extreme=force_extreme,
                    )
                    print(f"[DIRECTOR] shot={index} forcing camera+action mutation")
                elif attempts > 0:
                    shot_plan = director.mutate_plan(shot_plan, min_mutations=2)
                last_plan = dict(shot_plan)
                mode = shot_plan.get("shot_type", mode)
                prompt_base = beat.get("prompt") or beat.get("text") or f"{environment}:{mode}"
                prompt = (
                    f"{prompt_base} | camera {shot_plan['camera']} | shot {shot_plan['shot_type']}"
                    f" | action {shot_plan['action_bias']} | environment {shot_plan['environment_bias']}"
                    f" | cinematic variation {seed_value % 7}"
                )
                cache_payload = {
                    "phase": "phase25",
                    "episode_id": job_id,
                    "shot_index": index,
                    "seed": seed_value,
                    "prompt": prompt,
                    "duration_s": duration_s,
                    "fps": 30,
                    "w": 1280,
                    "h": 720,
                    "environment": environment,
                    "mode": mode,
                    "render_preset": os.getenv("MONEYOS_RENDER_PRESET", "fast_proof"),
                    "model": os.getenv("MONEYOS_ANIME3D_STYLE_PRESET", "key_art"),
                    "uniq": f"{job_id}:{index}:{seed_value}",
                }
                print(f"[SHOT_UNIQUENESS] payload_keys={sorted(cache_payload.keys())}")
                clip_id, cache_payload = compute_clip_id(cache_payload)
                if not clip_id or not isinstance(clip_id, str):
                    raise RuntimeError(
                        f"Missing clip_id for shot_index={index} payload={cache_payload}"
                    )
                cache_hash = clip_id.split("c_", 1)[-1]
                clip_dir = clips_root / clip_id
                ensure_dir(clip_dir)
                clip_path = clip_dir / "clip.mp4"
                final_mp4_path, telemetry = generate_clip_with_telemetry(
                    seconds=duration_s,
                    backend=os.getenv("MONEYOS_VISUAL_BACKEND", "hybrid"),
                    environment=environment,
                    mode=mode,
                    seed=str(seed_value),
                    cache_payload=cache_payload,
                )
                final_mp4_path = Path(final_mp4_path)
                if not final_mp4_path.exists():
                    raise RuntimeError(f"Missing rendered clip for shot_index={index}: {final_mp4_path}")
                src = Path(final_mp4_path).resolve()
                dst = Path(clip_path).resolve()
                if src != dst:
                    shutil.copy2(str(src), str(dst))
                else:
                    print(f"[PHASE25_COPY_SKIP] src==dst {dst}")
                clip_md5 = md5_file(str(clip_path))
                assert isinstance(clip_md5, str) and clip_md5
                print(
                    "[SHOT_UNIQUENESS] "
                    f"shot={index} attempt={attempts + 1} seed={seed_value} "
                    f"clip_id={clip_id} md5={clip_md5}"
                )
                if clip_md5 in seen_md5:
                    attempts += 1
                    seed_value += 123457
                    print(
                        "[SHOT_UNIQUENESS] "
                        f"shot={index} duplicate md5={clip_md5} reroll seed={seed_value} attempt={attempts}"
                    )
                    continue
                seen_md5.add(clip_md5)
                frames = extract_frames_ffmpeg(clip_path, num_frames=6)
                embeddings = embed_frames_with_clip(frames)
                is_duplicate, similarity_stats = compare_against_accepted(
                    embeddings,
                    accepted_embeddings,
                    thresholds=similarity_memory.get_thresholds(),
                )
                similarity_log = format_similarity_log(similarity_stats)
                if is_duplicate:
                    print(f"[AI_SIMILARITY] shot={index} {similarity_log} → DUPLICATE")
                    similarity_memory.record_reject(similarity_stats, reason="duplicate")
                    adjustment = similarity_memory.adjust_thresholds()
                    if adjustment:
                        print(
                            f"[LEARNING] {adjustment} mean_threshold → "
                            f"{similarity_memory.thresholds.mean:.3f}"
                        )
                    similarity_memory.save()
                    attempts += 1
                    seed_value += 123457
                    continue
                print(f"[AI_SIMILARITY] shot={index} {similarity_log} → PASS")
                diversity = score_diversity(embeddings, frames)
                diversity_score = diversity.get("diversity_score", 0.0)
                if diversity_score < diversity_threshold:
                    print(f"[DIVERSITY] shot={index} score={diversity_score:.2f} → REJECTED")
                    attempts += 1
                    seed_value += 123457
                    if attempts >= max_attempts_per_shot and not drop_used and accepted:
                        drop_used = True
                        lowest = min(accepted, key=lambda item: item.get("diversity_score", 1.0))
                        print(
                            "[DIVERSITY] "
                            f"dropping lowest-diversity clip={lowest.get('clip_id')}"
                        )
                        accepted.remove(lowest)
                        clips.remove(Path(lowest["clip_path"]))
                        accepted_embeddings[:] = [
                            item for item in accepted_embeddings if item.get("clip_id") != lowest.get("clip_id")
                        ]
                        recent_meta[:] = [
                            item for item in recent_meta if item.get("clip_id") != lowest.get("clip_id")
                        ]
                        similarity_memory.accepted = [
                            item
                            for item in similarity_memory.accepted
                            if item.get("clip_id") != lowest.get("clip_id")
                        ]
                        similarity_memory.save()
                        attempts = max_attempts_per_shot - 2
                    continue
                print(f"[DIVERSITY] shot={index} score={diversity_score:.2f} → PASS")
                meta = {
                    "phase": "phase25",
                    "clip_id": clip_id,
                    "shot_index": index,
                    "seed": seed_value,
                    "duration_s": float(beat.get("seconds", 0) or duration_s),
                    "prompt": prompt,
                    "shot_plan": shot_plan,
                    "clip_md5": clip_md5,
                    "source_final_path": str(final_mp4_path),
                    "cache_payload": cache_payload,
                    "diversity_score": diversity_score,
                    "motion_score": diversity.get("motion_score"),
                    "scene_score": diversity.get("scene_score"),
                }
                meta_path = clip_dir / "meta.json"
                meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
                print(f"[PHASE25_CLIP] clip={clip_path} meta={meta_path} md5={clip_md5}")
                accepted.append(
                    {
                        "clip_id": clip_id,
                        "clip_path": str(clip_path),
                        "diversity_score": diversity_score,
                    }
                )
                recent_meta.append({"clip_id": clip_id, "shot_plan": shot_plan})
                accepted_embeddings.append({"clip_id": clip_id, "embeddings": embeddings})
                similarity_memory.record_accept(clip_id, embeddings, similarity_stats, diversity)
                adjustment = similarity_memory.adjust_thresholds()
                if adjustment:
                    print(
                        f"[LEARNING] {adjustment} mean_threshold → "
                        f"{similarity_memory.thresholds.mean:.3f}"
                    )
                similarity_memory.save()
                clips.append(clip_path)
                print(
                    "[SHOT_UNIQUENESS] "
                    f"shot={index} seed={seed_value} clip_id={clip_id} md5={clip_md5} accepted"
                )
                with _jobs_lock:
                    _last_clip_telemetry.update(telemetry)
                _set_status(
                    job_id,
                    f"Rendered clip {index}/{clips_expected}",
                    stage_key="visuals",
                    progress_pct=min(95, int((index / max(1, clips_expected)) * 90)),
                    extra={
                        "clips_done": index,
                        "backend_used_last_clip": telemetry.get("backend_used"),
                        "gpu_used_last_clip": telemetry.get("gpu_used"),
                        "peak_vram_mb_last_clip": telemetry.get("peak_vram_mb"),
                        "base_visual_path_last_clip": telemetry.get("base_visual_path"),
                        "base_visual_validator_result": telemetry.get("base_visual_validator_result"),
                    },
                )
                break
            else:
                raise RuntimeError(
                    "Clip rejected after retries: "
                    f"shot_index={index} hashes={sorted(seen_md5)}"
                )
        if phase_env in {"phase25", "production"}:
            concat_path = tmp_root / "concat.txt"
            concat_lines = [f"file '{Path(item['clip_path']).as_posix()}'" for item in accepted]
            concat_path.write_text("\n".join(concat_lines), encoding="utf-8")
            output_path = phase25_root / "out_phase25.mp4"
            run_ffmpeg(
                [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(concat_path),
                    "-fflags",
                    "+genpts",
                    "-avoid_negative_ts",
                    "make_zero",
                    "-c:v",
                    "h264_nvenc",
                    "-pix_fmt",
                    "yuv420p",
                    "-r",
                    "30",
                    "-movflags",
                    "+faststart",
                    str(output_path),
                ]
            )
            print(f"[CONCAT] segments={len(concat_lines)} output={output_path}")
        else:
            output_path = safe_join("p2", "episodes", "episode_001.mp4")
            audio_path = safe_join("p2", "tmp", "episode_silence.wav")
            create_silent_audio(target_seconds, audio_path)
            assemble_episode(clips, audio_path, output_path)
            print(f"[CONCAT] segments={len(clips)} output={output_path}")
        produced_seconds = get_video_duration_seconds(output_path)
        if phase_env in {"phase25", "production"} and abs(produced_seconds - target_seconds) > 0.2:
            raise RuntimeError(
                f"Episode duration mismatch: produced {produced_seconds:.2f}s, "
                f"expected {target_seconds:.2f}s."
            )
        enforce_duration_contract(produced_seconds, target_seconds)
        with _jobs_lock:
            _last_job_snapshot.update(
                {
                    "job_type": "anime-episode-10m-hybrid",
                    "phase": phase_env,
                    "target_seconds": target_seconds,
                    "produced_seconds": produced_seconds,
                    "clips_expected": clips_expected,
                    "clips_done": clips_expected,
                    "last_output_path": str(output_path),
                }
            )
        _set_status(
            job_id,
            "Complete",
            stage_key="done",
            progress_pct=100,
            extra={
                "video_path": str(output_path),
                "duration": produced_seconds,
                "target_seconds": target_seconds,
                "clips_expected": clips_expected,
                "clips_done": clips_expected,
                "job_type": "anime-episode-10m-hybrid",
                "phase": phase_env,
            },
        )
    except Exception as exc:  # noqa: BLE001
        _set_error(job_id, f"Error: {exc}")


@app.post("/generate")
async def generate() -> JSONResponse:
    job_id = uuid.uuid4().hex
    _set_status(job_id, STATUS_SCRIPT)
    thread = threading.Thread(target=_run_job, args=(job_id,), daemon=True)
    thread.start()
    return JSONResponse({"job_id": job_id})


@app.post("/jobs/anime-episode-10m")
async def generate_anime_episode(
    req: AnimeEpisodeRequest = Body(default=AnimeEpisodeRequest()),
) -> JSONResponse:
    job_id = uuid.uuid4().hex
    _set_status(job_id, "Generating anime episode...")
    thread = threading.Thread(
        target=_run_anime_episode,
        args=(job_id, req.topic_hint, req.lane),
        daemon=True,
    )
    thread.start()
    return JSONResponse(
        {
            "job_id": job_id,
            "queued": False,
            "endpoint": "anime-episode-10m",
        }
    )


@app.post("/jobs/anime-episode-10m-hybrid")
async def generate_anime_episode_hybrid() -> JSONResponse:
    job_id = uuid.uuid4().hex
    phase_env, resolved_target, _raw_phase = _resolve_phase_target_seconds()
    _set_status(
        job_id,
        "Queued hybrid episode",
        stage_key="script",
        progress_pct=1,
        extra={
            "job_type": "anime-episode-10m-hybrid",
            "target_seconds": resolved_target,
            "phase": phase_env,
        },
    )
    print(f"[STATUS] expected_seconds={resolved_target:.2f} phase={phase_env}")
    thread = threading.Thread(target=_run_hybrid_episode, args=(job_id, resolved_target), daemon=True)
    thread.start()
    return JSONResponse({"job_id": job_id, "queued": False, "endpoint": "anime-episode-10m-hybrid"})


@app.post("/jobs/anime-episode-10m-autopilot")
async def enqueue_anime_episode_autopilot(
    req: AnimeEpisodeRequest = Body(default=AnimeEpisodeRequest()),
) -> JSONResponse:
    job_id = uuid.uuid4().hex
    _set_status(job_id, "Queued (autopilot)")
    autopilot_enqueue(job_id, topic_hint=req.topic_hint, lane=req.lane)
    return JSONResponse(
        {
            "job_id": job_id,
            "queued": True,
            "endpoint": "anime-episode-10m-autopilot",
        }
    )


@app.post("/jobs/anime-episode-60s-3d")
async def generate_anime_episode_3d_60s(
    req: Anime3DRequest = Body(default=Anime3DRequest()),
) -> JSONResponse:
    if VISUAL_MODE != "anime_3d":
        raise HTTPException(status_code=400, detail="MONEYOS_VISUAL_MODE must be anime_3d")
    from app.core.visuals.anime_3d.render_pipeline import _ensure_assets  # noqa: WPS433

    try:
        assets_root = get_assets_root()
        required_assets = {
            "characters/hero.blend": (assets_root / "characters" / "hero.blend"),
            "characters/enemy.blend": (assets_root / "characters" / "enemy.blend"),
            "envs/city.blend": (assets_root / "envs" / "city.blend"),
            "anims/idle.fbx": (assets_root / "anims" / "idle.fbx"),
            "anims/run.fbx": (assets_root / "anims" / "run.fbx"),
            "anims/punch.fbx": (assets_root / "anims" / "punch.fbx"),
            "vfx/explosion.png": (assets_root / "vfx" / "explosion.png"),
            "vfx/energy_arc.png": (assets_root / "vfx" / "energy_arc.png"),
            "vfx/smoke.png": (assets_root / "vfx" / "smoke.png"),
        }
        missing = [key for key, path in required_assets.items() if not path.exists()]
        strict_env = os.getenv("MONEYOS_STRICT_ASSETS") == "1"
        strict_req = bool(req.strict_assets) if req.strict_assets is not None else False
        explicit_strict = strict_env or strict_req
        strict_assets = 1 if explicit_strict else 0
        _ensure_assets(missing, strict_assets)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    job_id = uuid.uuid4().hex
    _set_status(job_id, "Queued 3D render")
    thread = threading.Thread(target=_run_anime_3d_60s, args=(job_id, req), daemon=True)
    thread.start()
    output_dir = anime_3d_output_dir(job_id)
    return JSONResponse(
        {
            "job_id": job_id,
            "output_dir": str(output_dir.resolve()),
            "final_video": str((output_dir / "final.mp4").resolve()),
        }
    )


@app.post("/jobs/anime-clip-3d")
async def generate_anime_clip_3d(
    req: Anime3DRequest = Body(default=Anime3DRequest()),
) -> JSONResponse:
    job_id = uuid.uuid4().hex
    _set_status(job_id, "Queued 3D clip")
    thread = threading.Thread(target=_run_anime_3d_clip, args=(job_id, req), daemon=True)
    thread.start()
    return JSONResponse({"job_id": job_id})


@app.post("/jobs/anime-episode-60s-3d/finalize")
async def finalize_anime_episode_3d(job_id: str = Body(..., embed=True)) -> JSONResponse:
    if VISUAL_MODE != "anime_3d":
        raise HTTPException(status_code=400, detail="MONEYOS_VISUAL_MODE must be anime_3d")
    try:
        result = finalize_anime_3d(job_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _set_status(job_id, "Complete", anime_3d_result=result, stage_key="done", progress_pct=100)
    return JSONResponse({"status": "ok", "job_id": job_id})


@app.get("/status/{job_id}")
async def status(job_id: str) -> JSONResponse:
    with _jobs_lock:
        data = _jobs.get(job_id)
    if not data:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(data)


@app.get("/events/{job_id}")
async def events(job_id: str) -> StreamingResponse:
    async def event_stream():
        while True:
            with _jobs_lock:
                data = _jobs.get(job_id)
            if not data:
                yield "event: error\ndata: {\"error\": \"Job not found\"}\n\n"
                return
            payload = json.dumps(data)
            yield f"event: status\ndata: {payload}\n\n"
            if data.get("success") or (data.get("status") or "").startswith("Error"):
                return
            await asyncio.sleep(1)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/videos/{filename}")
async def get_video(filename: str) -> FileResponse:
    path = VIDEO_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(path)


class HarvestRequest(BaseModel):
    query: str
    style: Optional[str] = None
    count: int = 10


@app.post("/assets/harvest")
async def harvest(req: HarvestRequest) -> JSONResponse:
    report = harvest_assets(req.query, req.style or "anime", req.count)
    return JSONResponse({"candidates": report.candidates, "selected": report.selected})


@app.get("/assets/harvest/report")
async def harvest_report() -> JSONResponse:
    paths = get_cache_paths()
    if not paths.report_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    return JSONResponse(json.loads(paths.report_path.read_text(encoding="utf-8")))


@app.get("/assets/characters/auto")
async def auto_characters() -> JSONResponse:
    if not AUTO_CHARACTERS_DIR.exists():
        return JSONResponse({"characters": []})
    characters = [path.name for path in AUTO_CHARACTERS_DIR.iterdir() if path.is_dir()]
    return JSONResponse({"characters": characters})


class UseCharactersRequest(BaseModel):
    character_ids: list[str]


@app.post("/assets/characters/auto/use")
async def use_auto_characters(req: UseCharactersRequest) -> JSONResponse:
    selected = []
    for character_id in req.character_ids:
        source = AUTO_CHARACTERS_DIR / character_id
        target = CHARACTERS_DIR / character_id
        if source.exists() and source.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            selected.append(character_id)
    return JSONResponse({"selected": selected})


@app.get("/assets/status")
async def assets_status() -> JSONResponse:
    characters = [path.name for path in CHARACTERS_DIR.iterdir()] if CHARACTERS_DIR.exists() else []
    auto_characters = [path.name for path in AUTO_CHARACTERS_DIR.iterdir()] if AUTO_CHARACTERS_DIR.exists() else []
    index_path = rebuild_animation_library()
    clips = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else []
    missing = []
    for required in ("walk", "talk", "punch"):
        if not any(clip.get("motion_type") == required for clip in clips):
            missing.append(required)
    return JSONResponse(
        {
            "characters": characters,
            "auto_characters": auto_characters,
            "clips_indexed": len(clips),
            "missing_categories": missing,
        }
    )
