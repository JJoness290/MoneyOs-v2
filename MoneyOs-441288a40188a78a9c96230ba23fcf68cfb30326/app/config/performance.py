from __future__ import annotations

import os
from math import floor

import psutil

_RAM_MODE_CACHE: str | None = None
_RAM_MODE_LOGGED = False

MAX_CPU_PERCENT = int(os.getenv("MONEYOS_MAX_CPU", "80"))
MAX_RAM_PERCENT = int(os.getenv("MONEYOS_MAX_RAM", "80"))
MAX_GPU_PERCENT = int(os.getenv("MONEYOS_MAX_GPU", "80"))
CHECK_INTERVAL_SEC = float(os.getenv("MONEYOS_CHECK_INTERVAL", "0.75"))


def ffmpeg_threads() -> int:
    if ram_mode() == "low":
        return 1
    logical_threads = psutil.cpu_count(logical=True) or 1
    return max(1, min(12, floor(logical_threads * 0.8)))


def total_ram_gb() -> float:
    return psutil.virtual_memory().total / (1024**3)


def ram_mode() -> str:
    global _RAM_MODE_CACHE, _RAM_MODE_LOGGED
    if _RAM_MODE_CACHE:
        return _RAM_MODE_CACHE
    env_mode = os.getenv("MONEYOS_RAM_MODE")
    if env_mode:
        env_mode = env_mode.strip().lower()
        if env_mode in {"low", "normal"}:
            _RAM_MODE_CACHE = env_mode
        else:
            _RAM_MODE_CACHE = "normal"
    else:
        _RAM_MODE_CACHE = "low" if total_ram_gb() <= 16 else "normal"
    if not _RAM_MODE_LOGGED:
        _RAM_MODE_LOGGED = True
        logical_threads = psutil.cpu_count(logical=True) or 1
        default_threads = max(1, min(12, floor(logical_threads * 0.8)))
        threads = 1 if _RAM_MODE_CACHE == "low" else default_threads
        print(
            f"[PERF] ram_mode={_RAM_MODE_CACHE} total_ram_gb={total_ram_gb():.1f} "
            f"ffmpeg_threads={threads}"
        )
    return _RAM_MODE_CACHE


def segment_workers() -> int:
    if ram_mode() == "low":
        return 1
    env_workers = os.getenv("MONEYOS_SEGMENT_WORKERS")
    if env_workers:
        try:
            return max(1, int(env_workers))
        except ValueError:
            return 1
    logical_threads = psutil.cpu_count(logical=True) or 1
    return max(1, min(4, floor(logical_threads * 0.5)))
