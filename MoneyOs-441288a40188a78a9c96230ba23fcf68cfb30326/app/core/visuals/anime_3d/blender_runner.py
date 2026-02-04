from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from app.config import BLENDER_ENGINE, BLENDER_GPU, BLENDER_PATH


@dataclass(frozen=True)
class BlenderCommand:
    script_path: Path
    args: list[str]


def _resolve_blender_path() -> Path:
    if BLENDER_PATH:
        return Path(BLENDER_PATH)
    candidates = [
        Path("C:/Program Files/Blender Foundation/Blender/blender.exe"),
        Path("C:/Program Files/Blender Foundation/Blender 4.0/blender.exe"),
        Path("C:/Program Files/Blender Foundation/Blender 3.6/blender.exe"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Blender not found. Set MONEYOS_BLENDER_PATH to blender.exe (e.g. C:/Program Files/Blender Foundation/Blender/blender.exe)."
    )


def build_blender_command(script_path: Path, args: Iterable[str]) -> list[str]:
    blender_path = _resolve_blender_path()
    engine = BLENDER_ENGINE
    gpu_flag = "1" if BLENDER_GPU else "0"
    return [
        str(blender_path),
        "-b",
        "--python",
        str(script_path),
        "--",
        "--engine",
        engine,
        "--gpu",
        gpu_flag,
        *list(args),
    ]


def run_blender(command: BlenderCommand) -> subprocess.CompletedProcess[str]:
    full_command = build_blender_command(command.script_path, command.args)
    return subprocess.run(
        full_command,
        check=True,
        text=True,
        capture_output=True,
    )
