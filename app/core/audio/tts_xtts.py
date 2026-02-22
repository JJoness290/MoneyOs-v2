from __future__ import annotations

import io
import json
import os
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class MoneyOSValidationError(RuntimeError):
    """Raised when MoneyOS audio config is invalid."""


@dataclass
class XTTSHandle:
    model: object | None
    device: str
    backend: str


def resolve_tts_license_mode() -> str:
    return os.getenv("MONEYOS_TTS_LICENSE", "none").strip().lower()


def configure_xtts_runtime_env(cache_root: Path) -> dict[str, str]:
    cache_root = Path(cache_root)
    tts_home = cache_root / "tts"
    xdg_cache_home = cache_root
    appdata_dir = cache_root / "appdata"
    tts_home.mkdir(parents=True, exist_ok=True)
    xdg_cache_home.mkdir(parents=True, exist_ok=True)
    appdata_dir.mkdir(parents=True, exist_ok=True)

    license_mode = resolve_tts_license_mode()
    if license_mode not in {"cpml", "commercial"}:
        raise MoneyOSValidationError(
            "MONEYOS_TTS_LICENSE must be set to 'cpml' (or 'commercial' if you have that entitlement) "
            "to run XTTS headlessly. Refusing to start interactive TTS prompt."
        )

    env_updates = {
        "TTS_HOME": str(tts_home),
        "XDG_CACHE_HOME": str(xdg_cache_home),
        "APPDATA": str(appdata_dir),
        "MONEYOS_TTS_LICENSE": license_mode,
    }
    if license_mode == "cpml":
        env_updates["COQUI_TOS_AGREED"] = "1"
    for key, value in env_updates.items():
        os.environ[key] = value
    return env_updates


def _resolve_device() -> str:
    desired = os.getenv("MONEYOS_TTS_DEVICE", "auto").strip().lower()
    if desired in {"cuda", "cpu"}:
        return desired
    try:
        import torch  # noqa: WPS433

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"




def _normalize_xtts_model_name(model_name: str) -> str:
    value = (model_name or "").strip()
    if not value:
        return "tts_models/multilingual/multi-dataset/xtts_v2"
    lowered = value.lower()
    alias_map = {
        "coqui/xtts-v2": "tts_models/multilingual/multi-dataset/xtts_v2",
        "xtts-v2": "tts_models/multilingual/multi-dataset/xtts_v2",
        "xtts_v2": "tts_models/multilingual/multi-dataset/xtts_v2",
    }
    if lowered in alias_map:
        return alias_map[lowered]
    parts = value.split("/")
    if len(parts) in {2, 3}:
        if lowered.endswith("xtts-v2") or lowered.endswith("xtts_v2"):
            return "tts_models/multilingual/multi-dataset/xtts_v2"
        raise MoneyOSValidationError(
            "Invalid MONEYOS_TTS_MODEL format. Expected either 'tts_models/<lang>/<dataset>/<model>' "
            "or xtts alias (coqui/XTTS-v2, xtts-v2)."
        )
    if len(parts) != 4:
        raise MoneyOSValidationError(
            "Invalid MONEYOS_TTS_MODEL format. Use e.g. tts_models/multilingual/multi-dataset/xtts_v2."
        )
    return value


def load_xtts(model_cache_dir: Path) -> XTTSHandle:
    env_updates = configure_xtts_runtime_env(model_cache_dir.parent if model_cache_dir.name == "tts" else model_cache_dir)
    model_cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"[VOICE] XTTS cache home set to {env_updates['TTS_HOME']}")
    print(f"[VOICE] XTTS model files will be downloaded to {env_updates['TTS_HOME']} if missing")
    device = _resolve_device()
    try:
        from TTS.api import TTS  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "XTTS backend unavailable. Install with: pip install TTS soundfile. "
            "Set MONEYOS_TTS_DEVICE=cpu if CUDA is missing."
        ) from exc

    configured_model = os.getenv("MONEYOS_TTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")
    model_name = _normalize_xtts_model_name(configured_model)
    tts = TTS(model_name=model_name, progress_bar=False, gpu=(device == "cuda"))
    return XTTSHandle(model=tts, device=device, backend=model_name)


def _emotion_speed(emotion: str | None, speed: float) -> float:
    tag = (emotion or "").lower()
    if tag in {"hype", "excited", "energetic"}:
        return speed * 1.06
    if tag in {"serious", "calm", "somber"}:
        return speed * 0.96
    return speed


def synthesize(
    handle: XTTSHandle,
    text: str,
    speaker_wav: str | None = None,
    language: str = "en",
    emotion: str | None = None,
    speed: float = 1.0,
) -> bytes:
    styled_speed = _emotion_speed(emotion, speed)
    styled_text = text.replace(".", ". ")
    if handle.model is None:
        raise RuntimeError("XTTS model handle is not initialized")

    wav = handle.model.tts(
        text=styled_text,
        speaker_wav=speaker_wav,
        language=language,
        speed=styled_speed,
    )
    arr = np.asarray(wav, dtype=np.float32)
    arr = np.clip(arr, -1.0, 1.0)
    pcm = (arr * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def write_voice_meta(path: Path, handle: XTTSHandle, settings: dict) -> None:
    payload = {"backend": handle.backend, "device": handle.device, **settings}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
