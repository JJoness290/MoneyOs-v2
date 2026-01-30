from __future__ import annotations

import threading
import uuid
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from app.config import VIDEO_DIR
from app.core.bootstrap import ensure_dependencies
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


@app.on_event("startup")
def bootstrap_dependencies() -> None:
    ensure_dependencies()


def _format_mmss(seconds: float) -> str:
    total_seconds = int(round(seconds))
    minutes, secs = divmod(total_seconds, 60)
    return f"{minutes:02d}:{secs:02d}"


def _set_status(job_id: str, status: str, result: Optional[PipelineResult] = None) -> None:
    with _jobs_lock:
        payload = _jobs.setdefault(job_id, {"status": STATUS_IDLE})
        payload["status"] = status
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


def _set_error(job_id: str, message: str) -> None:
    with _jobs_lock:
        payload = _jobs.setdefault(job_id, {"status": STATUS_IDLE})
        payload["status"] = message
        payload["success"] = False


def _run_job(job_id: str) -> None:
    try:
        result = run_pipeline(lambda status: _set_status(job_id, status))
        _set_status(job_id, STATUS_DONE, result)
    except Exception as exc:  # noqa: BLE001
        _set_error(job_id, f"Error: {exc}")


@app.get("/")
async def ui() -> HTMLResponse:
    index_path = Path(__file__).parent / "ui" / "index.html"
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.get("/api/system/specs")
async def system_specs() -> JSONResponse:
    return JSONResponse(get_system_specs())


@app.post("/generate")
async def generate() -> JSONResponse:
    job_id = uuid.uuid4().hex
    _set_status(job_id, STATUS_SCRIPT)
    thread = threading.Thread(target=_run_job, args=(job_id,), daemon=True)
    thread.start()
    return JSONResponse({"job_id": job_id})


@app.get("/status/{job_id}")
async def status(job_id: str) -> JSONResponse:
    with _jobs_lock:
        data = _jobs.get(job_id)
    if not data:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(data)


@app.get("/videos/{filename}")
async def get_video(filename: str) -> FileResponse:
    path = VIDEO_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(path)
