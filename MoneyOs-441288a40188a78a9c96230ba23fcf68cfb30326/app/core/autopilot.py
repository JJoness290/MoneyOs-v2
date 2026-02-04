from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, time as dt_time

from app.config import performance
from app.core.anime_episode import generate_anime_episode_10m


@dataclass
class AutopilotJob:
    job_id: str
    status: str
    started_at: float | None = None
    completed_at: float | None = None


_queue_lock = threading.Lock()
_queue: list[AutopilotJob] = []
_active_jobs: dict[str, AutopilotJob] = {}
_autopilot_thread: threading.Thread | None = None


def enqueue(job_id: str) -> None:
    with _queue_lock:
        _queue.append(AutopilotJob(job_id=job_id, status="queued"))


def _run_window_allows() -> bool:
    window = os.getenv("MONEYOS_RUN_WINDOW", "").strip()
    if not window:
        return True
    try:
        start_str, end_str = window.split("-", 1)
        start = dt_time.fromisoformat(start_str)
        end = dt_time.fromisoformat(end_str)
        now = datetime.now().time()
    except ValueError:
        return True
    if start <= end:
        return start <= now <= end
    return now >= start or now <= end


def _max_active_jobs() -> int:
    env_max = os.getenv("MONEYOS_AUTOPILOT_MAX_ACTIVE_JOBS")
    if env_max:
        try:
            return max(1, int(env_max))
        except ValueError:
            return 1
    return 1 if performance.ram_mode() == "low" else 2


def _interval_minutes() -> float:
    try:
        return float(os.getenv("MONEYOS_AUTOPILOT_INTERVAL_MINUTES", "180"))
    except ValueError:
        return 180.0


def _worker(job: AutopilotJob) -> None:
    job.status = "running"
    job.started_at = time.time()
    try:
        generate_anime_episode_10m()
        job.status = "completed"
    except Exception as exc:  # noqa: BLE001
        job.status = f"failed: {exc}"
    job.completed_at = time.time()
    with _queue_lock:
        _active_jobs.pop(job.job_id, None)


def _loop() -> None:
    while True:
        if not _run_window_allows():
            time.sleep(30)
            continue
        with _queue_lock:
            while _queue and len(_active_jobs) < _max_active_jobs():
                job = _queue.pop(0)
                _active_jobs[job.job_id] = job
                thread = threading.Thread(target=_worker, args=(job,), daemon=True)
                thread.start()
        time.sleep(max(60.0, _interval_minutes() * 60.0))


def start_autopilot() -> None:
    global _autopilot_thread
    if _autopilot_thread and _autopilot_thread.is_alive():
        return
    if os.getenv("MONEYOS_AUTOPILOT", "0") != "1":
        return
    _autopilot_thread = threading.Thread(target=_loop, daemon=True)
    _autopilot_thread.start()


def status() -> dict:
    with _queue_lock:
        return {
            "queued": [job.job_id for job in _queue],
            "active": {job_id: job.status for job_id, job in _active_jobs.items()},
            "max_active": _max_active_jobs(),
        }
