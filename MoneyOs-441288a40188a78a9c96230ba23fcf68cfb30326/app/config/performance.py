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
    mode = ram_mode()
    logical_threads = psutil.cpu_count(logical=True) or 1
    default_threads = max(1, min(6, max(1, logical_threads - 2)))
    if mode == "low":
        return 1
    if mode == "high":
        return max(default_threads, min(12, logical_threads))
    return default_threads


def total_ram_gb() -> float:
    return psutil.virtual_memory().total / (1024**3)


def ram_mode() -> str:
    global _RAM_MODE_CACHE, _RAM_MODE_LOGGED
    if _RAM_MODE_CACHE:
        return _RAM_MODE_CACHE
    env_mode = os.getenv("MONEYOS_RAM_MODE")
    if env_mode:
        env_mode = env_mode.strip().lower()
        if env_mode in {"low", "balanced", "high"}:
            _RAM_MODE_CACHE = env_mode
        else:
            _RAM_MODE_CACHE = "balanced"
    else:
        _RAM_MODE_CACHE = "low" if total_ram_gb() <= 16 else "balanced"
    if not _RAM_MODE_LOGGED:
        _RAM_MODE_LOGGED = True
        logical_threads = psutil.cpu_count(logical=True) or 1
        default_threads = max(1, min(6, max(1, logical_threads - 2)))
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
