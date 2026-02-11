from __future__ import annotations

from dataclasses import dataclass
import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable

from app.config import OUTPUT_DIR


Logger = Callable[[str], None]


@dataclass(frozen=True)
class HardwareProfile:
    gpu_name: str | None
    vram_total_gb: float
    vram_free_gb: float | None
    ram_gb: float | None


def _log_default(message: str) -> None:
    print(f"[ANIME] {message}")


def detect_gpu(torch_module) -> dict[str, Any]:
    if not torch_module.cuda.is_available():
        return {"gpu_name": None, "vram_total_gb": 0.0, "vram_free_gb": None}
    props = torch_module.cuda.get_device_properties(0)
    total_gb = props.total_memory / (1024**3)
    free_gb = None
    try:
        free_gb = torch_module.cuda.mem_get_info()[0] / (1024**3)
    except Exception:  # noqa: BLE001
        free_gb = None
    return {"gpu_name": props.name, "vram_total_gb": total_gb, "vram_free_gb": free_gb}


def detect_ram_gb() -> float | None:
    try:
        import psutil  # noqa: WPS433

        return psutil.virtual_memory().total / (1024**3)
    except Exception:  # noqa: BLE001
        return None


def detect_hardware(torch_module) -> HardwareProfile:
    gpu = detect_gpu(torch_module)
    ram_gb = detect_ram_gb()
    return HardwareProfile(
        gpu_name=gpu["gpu_name"],
        vram_total_gb=float(gpu["vram_total_gb"]),
        vram_free_gb=gpu["vram_free_gb"],
        ram_gb=ram_gb,
    )


def sanitize_filename_component(value: str) -> str:
    keep = []
    for char in value:
        if char.isalnum() or char in {"_", "-", "."}:
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "unknown"


def _profile_dir() -> Path:
    override = os.getenv("MONEYOS_QUALITY_PROFILE_DIR")
    if override:
        return Path(override)
    return OUTPUT_DIR / "cache" / "quality_profiles"


def profile_key(hardware: HardwareProfile, torch_version: str, model_id: str) -> str:
    gpu_part = sanitize_filename_component(hardware.gpu_name or "cpu")
    vram_part = f"{hardware.vram_total_gb:.0f}GB"
    model_hash = hashlib.sha256(model_id.encode("utf-8")).hexdigest()[:8]
    torch_hash = hashlib.sha256(torch_version.encode("utf-8")).hexdigest()[:6]
    return f"{gpu_part}_{vram_part}_{torch_hash}_{model_hash}"


def load_cached_profile(key: str) -> dict[str, Any] | None:
    if os.getenv("MONEYOS_QUALITY_CACHE", "1") == "0":
        return None
    path = _profile_dir() / f"{key}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def save_cached_profile(key: str, profile: dict[str, Any]) -> None:
    if os.getenv("MONEYOS_QUALITY_CACHE", "1") == "0":
        return
    path = _profile_dir()
    path.mkdir(parents=True, exist_ok=True)
    (path / f"{key}.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")


def base_preset_for_vram(vram_gb: float, model_family: str) -> dict[str, Any]:
    if model_family == "sdxl":
        if vram_gb < 12:
            return {
                "width": 768,
                "height": 768,
                "steps": 12,
                "guidance": 5.5,
                "fp16": True,
                "cpu_offload": True,
                "attention_slicing": True,
                "vae_slicing": True,
                "xformers": False,
            }
        if vram_gb <= 16:
            return {
                "width": 1024,
                "height": 1024,
                "steps": 14,
                "guidance": 5.5,
                "fp16": True,
                "cpu_offload": True,
                "attention_slicing": True,
                "vae_slicing": True,
                "xformers": False,
            }
        return {
            "width": 1024,
            "height": 1024,
            "steps": 20,
            "guidance": 5.5,
            "fp16": True,
            "cpu_offload": False,
            "attention_slicing": False,
            "vae_slicing": False,
            "xformers": False,
        }
    if vram_gb <= 6:
        return {
            "width": 448,
            "height": 448,
            "steps": 12,
            "guidance": 6.0,
            "fp16": True,
            "cpu_offload": True,
            "attention_slicing": True,
            "vae_slicing": True,
            "xformers": False,
        }
    if vram_gb <= 8:
        return {
            "width": 512,
            "height": 512,
            "steps": 14,
            "guidance": 6.5,
            "fp16": True,
            "cpu_offload": True,
            "attention_slicing": True,
            "vae_slicing": True,
            "xformers": False,
        }
    if vram_gb <= 12:
        return {
            "width": 640,
            "height": 640,
            "steps": 20,
            "guidance": 6.8,
            "fp16": True,
            "cpu_offload": True,
            "attention_slicing": True,
            "vae_slicing": True,
            "xformers": False,
        }
    return {
        "width": 832,
        "height": 832,
        "steps": 28,
        "guidance": 7.0,
        "fp16": True,
        "cpu_offload": False,
        "attention_slicing": False,
        "vae_slicing": False,
        "xformers": False,
    }


def _ladder_for_mode(
    vram_gb: float,
    model_family: str,
    base_profile: dict[str, Any],
    quality_mode: str,
) -> list[dict[str, Any]]:
    width = base_profile["width"]
    steps = base_profile["steps"]
    if quality_mode == "fast":
        return [{"width": width, "height": width, "steps": max(10, int(steps * 0.8))}]

    ladder: list[dict[str, Any]] = [{"width": width, "height": width, "steps": steps}]
    if model_family == "sd15" and vram_gb <= 8:
        ladder = [
            {"width": 512, "height": 512, "steps": 14},
            {"width": 512, "height": 512, "steps": 18},
            {"width": 576, "height": 576, "steps": 16},
            {"width": 576, "height": 576, "steps": 20},
            {"width": 640, "height": 640, "steps": 16},
        ]
    elif model_family == "sd15" and vram_gb <= 12:
        ladder = [
            {"width": 640, "height": 640, "steps": 20},
            {"width": 704, "height": 704, "steps": 22},
            {"width": 768, "height": 768, "steps": 24},
        ]
    elif model_family == "sdxl":
        ladder = [
            {"width": width, "height": width, "steps": steps},
            {"width": min(1024, width + 128), "height": min(1024, width + 128), "steps": steps + 2},
        ]
    if quality_mode == "auto":
        return ladder[:3]
    return ladder


def autotune_sd_profile(
    pipeline,
    base_profile: dict[str, Any],
    quality_mode: str,
    logger: Logger | None = None,
    debug: bool = False,
) -> dict[str, Any]:
    log = logger or _log_default
    if quality_mode == "manual":
        return base_profile
    ladder = _ladder_for_mode(base_profile["vram_gb"], base_profile["model_family"], base_profile, quality_mode)
    prompt = "test anime still frame, high quality, sharp lineart"
    negative_prompt = "blurry, lowres, artifacts"
    best = base_profile
    for candidate in ladder:
        width = candidate["width"]
        height = candidate["height"]
        steps = candidate["steps"]
        if debug:
            log(f"[QUALITY] probe {width}x{height} steps={steps}")
        try:
            pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=base_profile["guidance"],
                generator=base_profile["generator"],
            )
            best = {**base_profile, **candidate}
        except Exception as exc:  # noqa: BLE001
            if "out of memory" in str(exc).lower():
                log(f"[QUALITY] OOM at {width}x{height} steps={steps}, stopping ladder")
                try:
                    import torch  # noqa: WPS433

                    torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001
                    pass
                gc.collect()
                break
            log(f"[QUALITY] probe failed: {exc}")
            break
    if quality_mode in {"auto", "max"}:
        try:
            pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=best["width"],
                height=best["height"],
                num_inference_steps=best["steps"],
                guidance_scale=best["guidance"],
                generator=base_profile["generator"],
            )
        except Exception as exc:  # noqa: BLE001
            log(f"[QUALITY] confirmation failed: {exc}")
    return best
