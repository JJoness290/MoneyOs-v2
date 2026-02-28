from __future__ import annotations

import gc
import json
import os
from pathlib import Path

from app.core.paths import get_cache_root


def is_oom_like_error(exc: Exception) -> bool:
    text = str(exc)
    lowered = text.lower()
    patterns = [
        "cuda out of memory",
        "cuda out of memory.",
        "cuda out of memory",
        "cudnn_status_not_supported",
        "hiperroroutofmemory",
    ]
    if any(p in lowered for p in patterns):
        return True
    try:
        import torch  # noqa: WPS433

        oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
        if oom_type is not None and isinstance(exc, oom_type):
            return True
    except Exception:
        pass
    return False


def _clamp_fraction(value: float) -> float:
    return max(0.50, min(0.90, value))


def _last_good_file() -> Path:
    return get_cache_root() / "stability" / "last_good_vram_fraction.json"


def load_last_good_fraction() -> float | None:
    path = _last_good_file()
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return _clamp_fraction(float(payload.get("vram_fraction")))
    except Exception:
        return None


def save_last_good_fraction(value: float) -> None:
    path = _last_good_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"vram_fraction": _clamp_fraction(value)}), encoding="utf-8")


def resolve_initial_fraction() -> float:
    raw = os.getenv("MONEYOS_VRAM_FRACTION", "").strip()
    if raw:
        try:
            return _clamp_fraction(float(raw))
        except ValueError:
            pass
    remembered = load_last_good_fraction()
    if remembered is not None:
        return remembered
    return 0.80


def build_fraction_ladder(initial: float, attempts_total: int = 6) -> list[float]:
    initial = _clamp_fraction(initial)
    base = [initial, 0.75, 0.70, 0.65, 0.60, 0.55]
    ladder: list[float] = []
    for value in base:
        v = _clamp_fraction(value)
        if v > initial + 1e-9:
            continue
        if v not in ladder:
            ladder.append(v)
        if len(ladder) >= attempts_total:
            break
    if not ladder:
        ladder = [initial]
    return ladder[:attempts_total]


def release_cuda_memory() -> None:
    gc.collect()
    try:
        import torch  # noqa: WPS433

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass
