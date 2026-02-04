from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from app.config import BLENDER_ENGINE, BLENDER_GPU, BLENDER_PATH


@dataclass(frozen=True)
class BlenderCommand:
    script_path: Path
    args: list[str]


@dataclass(frozen=True)
class BlenderDetection:
    found: bool
    path: str | None
    version: str | None
    error: str | None


def _candidate_paths() -> list[Path]:
    base_dir = Path("C:/Program Files/Blender Foundation")
    candidates = [
        base_dir / "Blender" / "blender.exe",
        base_dir / "Blender 4.0" / "blender.exe",
        base_dir / "Blender 3.6" / "blender.exe",
    ]
    if base_dir.exists():
        for path in sorted(base_dir.glob("Blender*/blender.exe")):
            candidates.append(path)
    return candidates


def _resolve_blender_path() -> Path:
    if BLENDER_PATH:
        return Path(BLENDER_PATH)
    for candidate in _candidate_paths():
        if candidate.exists():
            return candidate
    which_path = shutil.which("blender")
    if which_path:
        return Path(which_path)
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
        "-P",
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


def run_blender_capture(command: BlenderCommand) -> subprocess.CompletedProcess[str]:
    full_command = build_blender_command(command.script_path, command.args)
    return subprocess.run(
        full_command,
        check=False,
        text=True,
        capture_output=True,
    )


def detect_blender() -> BlenderDetection:
    try:
        path = _resolve_blender_path()
    except FileNotFoundError as exc:
        return BlenderDetection(found=False, path=None, version=None, error=str(exc))
    version = None
    try:
        result = subprocess.run(
            [str(path), "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
        output = (result.stdout or result.stderr).splitlines()
        if output:
            version = output[0].strip()
    except Exception as exc:  # noqa: BLE001
        return BlenderDetection(found=True, path=str(path), version=None, error=str(exc))
    return BlenderDetection(found=True, path=str(path), version=version, error=None)
