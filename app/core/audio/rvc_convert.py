from __future__ import annotations

import os
from pathlib import Path


def maybe_apply_rvc(input_wav: Path, output_wav: Path) -> Path:
    if os.getenv("MONEYOS_VOICE_CONVERT", "0") != "1":
        print("[VOICE] RVC disabled")
        return input_wav
    model_path = os.getenv("MONEYOS_RVC_MODEL_PATH", "").strip()
    if not model_path:
        print("[VOICE] RVC requested but MONEYOS_RVC_MODEL_PATH is empty; using XTTS output")
        return input_wav
    if not Path(model_path).exists():
        raise RuntimeError(f"RVC model not found: {model_path}. Set MONEYOS_RVC_MODEL_PATH to a local model path.")
    raise RuntimeError(
        "RVC conversion scaffold is enabled but runtime converter is not installed. "
        "Install your local RVC inference tool and wire this hook."
    )
