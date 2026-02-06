from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def _run(cmd: list[str]) -> tuple[int, str]:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        return 127, f"FileNotFoundError: {exc}"
    return result.returncode, (result.stdout or "") + (result.stderr or "")


def detect_ffmpeg() -> dict[str, object]:
    code, output = _run(["ffmpeg", "-hide_banner", "-encoders"])
    if code != 0:
        return {"available": False, "error": output.strip()}
    return {
        "available": True,
        "nvenc": "h264_nvenc" in output or "hevc_nvenc" in output,
    }


def detect_blender() -> dict[str, object]:
    blender_path = None
    for env_key in ("BLENDER_BINARY", "BLENDER_PATH"):
        env_value = os.getenv(env_key)
        if env_value and Path(env_value).exists():
            blender_path = env_value
            break
    if blender_path is None:
        default_path = Path(r"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe")
        if default_path.exists():
            blender_path = str(default_path)
    command = [blender_path, "--version"] if blender_path else ["blender", "--version"]
    code, output = _run(command)
    if code != 0:
        return {"available": False, "error": output.strip()}
    version = output.splitlines()[0] if output else ""
    return {"available": True, "version": version}


def detect_cuda() -> dict[str, object]:
    try:
        import torch  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": str(exc)}
    if not torch.cuda.is_available():
        return {"available": False, "error": "CUDA not available"}
    total, free = torch.cuda.mem_get_info()
    return {
        "available": True,
        "device": torch.cuda.get_device_name(0),
        "total_gb": round(total / (1024**3), 2),
        "free_gb": round(free / (1024**3), 2),
    }


def detect_ai_backend() -> dict[str, object]:
    backends = []
    try:
        import diffusers  # type: ignore
    except Exception:
        diffusers = None
    if diffusers is not None:
        backends.append("diffusers")
    return {"available": bool(backends), "backends": backends}


def capabilities_snapshot(cache_path: Path) -> dict[str, object]:
    snapshot = {
        "ffmpeg": detect_ffmpeg(),
        "blender": detect_blender(),
        "cuda": detect_cuda(),
        "ai_video": detect_ai_backend(),
        "short_workdir": os.getenv("MONEYOS_SHORT_WORKDIR", r"C:\\MoneyOS\\work"),
        "path_max": int(os.getenv("MONEYOS_PATH_MAX", "220")),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    return snapshot
