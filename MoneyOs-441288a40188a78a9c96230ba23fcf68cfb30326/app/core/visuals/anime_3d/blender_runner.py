from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from app.config import BLENDER_ENGINE, BLENDER_GPU, BLENDER_PATH
from src.utils.cli_args import validate_no_empty_value_flags


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


def build_blender_command(
    script_path: Path,
    args: Iterable[str],
    flags_requiring_value: Iterable[str] | None = None,
) -> list[str]:
    blender_path = _resolve_blender_path()
    engine = BLENDER_ENGINE
    gpu_flag = "1" if BLENDER_GPU else "0"
    if flags_requiring_value is None:
        flags_requiring_value = {"--character-asset", "--seed", "--fingerprint"}
    validate_no_empty_value_flags(args, flags_requiring_value)
    return [
        str(blender_path),
        "--background",
        "--factory-startup",
        "--python",
        str(script_path),
        "--",
        "--engine",
        engine,
        "--gpu",
        gpu_flag,
        *list(args),
    ]


def _extract_arg_value(args: Iterable[str], flag: str) -> str | None:
    args_list = list(args)
    for index, value in enumerate(args_list):
        if value == flag and index + 1 < len(args_list):
            return args_list[index + 1]
    return None


def _resolve_output_dir(args: Iterable[str]) -> Path:
    output_value = _extract_arg_value(args, "--output") or _extract_arg_value(args, "--report")
    if output_value:
        return Path(output_value).expanduser().resolve().parent
    return Path.cwd()


def _write_spawn_error(output_dir: Path, command: list[str], error: Exception) -> None:
    payload = {"stage": "spawn", "error": str(error), "cmd": command}
    (output_dir / "blender_error.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_blender(command: BlenderCommand) -> subprocess.CompletedProcess[str]:
    full_command = build_blender_command(command.script_path, command.args)
    output_dir = _resolve_output_dir(command.args)
    stdout_path = output_dir / "blender_stdout.txt"
    stderr_path = output_dir / "blender_stderr.txt"
    try:
        result = subprocess.run(
            full_command,
            check=False,
            text=True,
            capture_output=True,
        )
    except Exception as exc:  # noqa: BLE001
        _write_spawn_error(output_dir, full_command, exc)
        raise
    stdout_path.write_text(result.stdout or "", encoding="utf-8")
    stderr_path.write_text(result.stderr or "", encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            f"Blender failed with code {result.returncode}. See {stderr_path} for details."
        )
    return result


def run_blender(command: BlenderCommand) -> subprocess.CompletedProcess[str]:
    return _run_blender(command)


def run_blender_capture(command: BlenderCommand) -> subprocess.CompletedProcess[str]:
    return _run_blender(command)


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
