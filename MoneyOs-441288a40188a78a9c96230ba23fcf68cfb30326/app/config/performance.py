from __future__ import annotations

import os
from math import floor

import psutil

MAX_CPU_PERCENT = int(os.getenv("MONEYOS_MAX_CPU", "80"))
MAX_RAM_PERCENT = int(os.getenv("MONEYOS_MAX_RAM", "80"))
MAX_GPU_PERCENT = int(os.getenv("MONEYOS_MAX_GPU", "80"))
CHECK_INTERVAL_SEC = float(os.getenv("MONEYOS_CHECK_INTERVAL", "0.75"))


def ffmpeg_threads() -> int:
    logical_threads = psutil.cpu_count(logical=True) or 1
    return max(1, min(12, floor(logical_threads * 0.8)))
