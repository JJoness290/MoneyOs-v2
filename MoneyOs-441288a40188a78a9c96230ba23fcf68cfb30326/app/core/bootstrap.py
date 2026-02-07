from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from typing import Iterable

from app.config import performance
from app.core.visuals.ffmpeg_utils import encoder_self_check, has_nvenc


_REQUIRED_PACKAGES = {
    "diffusers": "0.30.3",
    "transformers": "4.44.2",
    "accelerate": "0.33.0",
    "safetensors": "0.4.4",
    "pillow": "10.4.0",
    "numpy": "2.0.1",
    "tiktoken": "0.7.0",
    "protobuf": "5.27.3",
}


def _log(message: str) -> None:
    print(f"[BOOTSTRAP] {message}")


def _in_virtualenv() -> bool:
    return sys.prefix != sys.base_prefix


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _pip_install(args: Iterable[str]) -> bool:
    command = [sys.executable, "-m", "pip", "install", *args]
    _log(f"Running: {' '.join(command)}")
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        _log(f"pip install failed with exit code {result.returncode}")
        return False
    return True


def ensure_dependencies() -> None:
    if os.getenv("MONEYOS_AUTO_PIP", "1") == "0":
        _log("Auto-install disabled via MONEYOS_AUTO_PIP=0")
        return

    from app.core.paths import get_assets_root, get_output_root

    _log(f"assets_root={get_assets_root()}")
    _log(f"output_root={get_output_root()}")
    _log(f"Virtualenv detected={_in_virtualenv()}")
    missing = []
    for module, version in _REQUIRED_PACKAGES.items():
        if not _has_module(module):
            missing.append(f"{module}=={version}")
    if missing:
        _log(f"Installing missing packages: {', '.join(missing)}")
        _pip_install(missing)
    else:
        _log("All required packages already installed.")

    if not _has_module("torch"):
        try:
            nvenc = has_nvenc()
        except Exception as exc:  # noqa: BLE001
            nvenc = f"error={exc}"
        _log(f"GPU check: cuda_available=False ffmpeg_nvenc={nvenc}")
        _log(
            "Torch is not installed. Install it manually from https://pytorch.org/get-started/locally/ "
            "to enable anime visuals."
        )
        _log(
            "Runtime settings: "
            f"MONEYOS_USE_GPU={os.getenv('MONEYOS_USE_GPU', '0')} "
            f"MONEYOS_NVENC_CODEC={os.getenv('MONEYOS_NVENC_CODEC', 'h264')} "
            f"MONEYOS_NVENC_QUALITY={os.getenv('MONEYOS_NVENC_QUALITY', 'balanced')} "
            f"MONEYOS_NVENC_MODE={os.getenv('MONEYOS_NVENC_MODE', 'cq')} "
            f"MONEYOS_RAM_MODE={performance.ram_mode()}"
        )
        try:
            encoder_status = encoder_self_check()
        except Exception as exc:  # noqa: BLE001
            encoder_status = f"error={exc}"
        _log(f"Encoder check: {encoder_status}")
        return
    _log("Torch already installed.")
    import torch  # noqa: WPS433

    cuda_available = torch.cuda.is_available()
    gpu_name = "Unknown GPU"
    if cuda_available:
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:  # noqa: BLE001
            gpu_name = "Unknown GPU"
    _log(
        "Torch status "
        f"torch={torch.__version__} "
        f"cuda_available={cuda_available} "
        f"gpu={gpu_name}"
    )
    try:
        nvenc = has_nvenc()
    except Exception as exc:  # noqa: BLE001
        nvenc = f"error={exc}"
    _log(f"GPU check: cuda_available={cuda_available} ffmpeg_nvenc={nvenc}")
    if os.getenv("MONEYOS_USE_GPU", "0") == "1" and not cuda_available:
        _log(
            "MONEYOS_USE_GPU=1 but CUDA is unavailable. Install a CUDA-enabled torch build "
            "from https://pytorch.org/get-started/locally/."
        )
    _log(
        "Runtime settings: "
        f"MONEYOS_USE_GPU={os.getenv('MONEYOS_USE_GPU', '0')} "
        f"MONEYOS_NVENC_CODEC={os.getenv('MONEYOS_NVENC_CODEC', 'h264')} "
        f"MONEYOS_NVENC_QUALITY={os.getenv('MONEYOS_NVENC_QUALITY', 'balanced')} "
        f"MONEYOS_NVENC_MODE={os.getenv('MONEYOS_NVENC_MODE', 'cq')} "
        f"MONEYOS_RAM_MODE={performance.ram_mode()}"
    )
    try:
        encoder_status = encoder_self_check()
    except Exception as exc:  # noqa: BLE001
        encoder_status = f"error={exc}"
    _log(f"Encoder check: {encoder_status}")
