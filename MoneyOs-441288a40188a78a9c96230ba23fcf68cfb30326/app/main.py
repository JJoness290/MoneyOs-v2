from __future__ import annotations

import json
import threading
import asyncio
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

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
    render_anime_3d_60s,
)
from app.core.pipeline import PipelineResult, run_pipeline
from app.core.system_specs import get_system_specs

app = FastAPI()

STATUS_IDLE = "Idle"
STATUS_SCRIPT = "Generating script..."
STATUS_BROLL = "Downloading B-roll..."
STATUS_RENDER = "Rendering video..."
STATUS_DONE = "Done"

_jobs_lock = threading.Lock()
_jobs: Dict[str, dict] = {}
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


@app.on_event("startup")
def bootstrap_dependencies() -> None:
    ensure_dependencies()
    start_autopilot()


def _format_mmss(seconds: float) -> str:
    total_seconds = int(round(seconds))
    minutes, secs = divmod(total_seconds, 60)
    return f"{minutes:02d}:{secs:02d}"


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


def _run_anime_3d_60s(job_id: str) -> None:
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
        result = render_anime_3d_60s(job_id, status_callback=_update)
        _set_status(job_id, "Complete", anime_3d_result=result, stage_key="done", progress_pct=100)
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
        "anime_3d_ready": blender.found and VISUAL_MODE == "anime_3d",
        "last_error": blender.error,
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
    req: AnimeEpisodeRequest = Body(default=AnimeEpisodeRequest()),
) -> JSONResponse:
    _ = req
    if VISUAL_MODE != "anime_3d":
        raise HTTPException(status_code=400, detail="MONEYOS_VISUAL_MODE must be anime_3d")
    from app.core.visuals.anime_3d.render_pipeline import _ensure_assets  # noqa: WPS433

    try:
        _ensure_assets()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    job_id = uuid.uuid4().hex
    _set_status(job_id, "Queued 3D render")
    thread = threading.Thread(target=_run_anime_3d_60s, args=(job_id,), daemon=True)
    thread.start()
    output_dir = anime_3d_output_dir(job_id)
    return JSONResponse(
        {
            "job_id": job_id,
            "output_dir": str(output_dir.resolve()),
            "final_video": str((output_dir / "final.mp4").resolve()),
        }
    )


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
