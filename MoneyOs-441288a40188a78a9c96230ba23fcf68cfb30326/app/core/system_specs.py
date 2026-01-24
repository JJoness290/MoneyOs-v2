from __future__ import annotations

import os
import platform
import subprocess
from typing import Iterable

import psutil

from app.config import performance


def _run_command(command: list[str]) -> str:
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=2)
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip()


def _parse_wmic_output(output: str) -> list[str]:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return [line for line in lines if line.lower() != "name"]


def _get_cpu_model() -> str:
    model = platform.processor()
    if model:
        return model
    if os.name == "nt":
        wmic = _run_command(["wmic", "cpu", "get", "Name"])
        names = _parse_wmic_output(wmic)
        if names:
            return names[0]
    return platform.uname().processor or "Unknown"


def _get_gpu_names() -> list[str]:
    nvidia = _run_command(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    names = [line.strip() for line in nvidia.splitlines() if line.strip()]
    if names:
        return names
    if os.name == "nt":
        wmic = _run_command(["wmic", "path", "win32_VideoController", "get", "name"])
        return _parse_wmic_output(wmic)
    return []


def _bytes_to_gb(value: int) -> float:
    return round(value / (1024**3), 2)


def get_system_specs() -> dict:
    cpu_model = _get_cpu_model()
    physical_cores = psutil.cpu_count(logical=False) or 0
    logical_threads = psutil.cpu_count(logical=True) or 0
    total_ram_gb = _bytes_to_gb(psutil.virtual_memory().total)
    gpu_names = _get_gpu_names()
    return {
        "cpu_model": cpu_model,
        "physical_cores": physical_cores,
        "logical_threads": logical_threads,
        "total_ram_gb": total_ram_gb,
        "gpus": gpu_names,
        "os": platform.platform(),
        "python": platform.python_version(),
        "caps": {
            "cpu_percent": performance.MAX_CPU_PERCENT,
            "ram_percent": performance.MAX_RAM_PERCENT,
            "gpu_percent": performance.MAX_GPU_PERCENT,
        },
    }
