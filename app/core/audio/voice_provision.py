from __future__ import annotations

import base64
import os
from pathlib import Path

DEFAULT_WAV_BASE64 = (
    "UklGRlQAAABXQVZFZm10IBAAAAABAAEAIlYAAESsAAACABAAZGF0YTAAAACAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgA=="
)


def ensure_voice_ref_wav(request_path: str | None = None) -> Path:
    candidates: list[Path] = []
    if request_path:
        candidates.append(Path(request_path))
    env_ref = os.getenv("MONEYOS_VOICE_REF_WAV", "").strip()
    if env_ref:
        candidates.append(Path(env_ref))
    candidates.append(Path(r"C:\MoneyOS\assets\voices\default.wav"))
    for c in candidates:
        if c.exists():
            return c
    out = Path(r"C:\MoneyOS\assets\voices\default.wav")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(base64.b64decode(DEFAULT_WAV_BASE64))
    return out
