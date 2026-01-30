from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from typing import Iterable


_REQUIRED_PACKAGES = {
    "diffusers": "0.30.3",
    "transformers": "4.44.2",
    "accelerate": "0.33.0",
    "safetensors": "0.4.4",
    "pillow": "10.4.0",
    "numpy": "2.0.1",
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
        _log(
            "Torch is not installed. Install it manually from https://pytorch.org/get-started/locally/ "
            "to enable anime visuals."
        )
        return
    _log("Torch already installed.")
    if os.getenv("MONEYOS_USE_GPU", "0") == "1":
        try:
            import torch  # noqa: WPS433

            if not torch.cuda.is_available():
                _log(
                    "MONEYOS_USE_GPU=1 but CUDA is unavailable. Install a CUDA-enabled torch build "
                    "from https://pytorch.org/get-started/locally/."
                )
        except Exception as exc:  # noqa: BLE001
            _log(f"Unable to verify CUDA availability: {exc}")
