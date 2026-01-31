from __future__ import annotations

from dataclasses import dataclass
import gc
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Callable

from app.config import OUTPUT_DIR


Logger = Callable[[str], None]


@dataclass(frozen=True)
class HardwareProfile:
    gpu_name: str | None
    vram_total_gb: float
    vram_free_gb: float | None
    ram_gb: float | None
    cpu_cores: int | None


def _log_default(message: str) -> None:
    print(f"[QUALITY] {message}")


def detect_hardware(torch_module) -> HardwareProfile:
    gpu_name = None
    vram_total_gb = 0.0
    vram_free_gb = None
    if torch_module.cuda.is_available():
        props = torch_module.cuda.get_device_properties(0)
        gpu_name = props.name
        vram_total_gb = props.total_memory / (1024**3)
        try:
            vram_free_gb = torch_module.cuda.mem_get_info()[0] / (1024**3)
        except Exception:  # noqa: BLE001
            vram_free_gb = None
    ram_gb = None
    cpu_cores = None
    try:
        import psutil  # noqa: WPS433

        ram_gb = psutil.virtual_memory().total / (1024**3)
        cpu_cores = psutil.cpu_count(logical=True)
    except Exception:  # noqa: BLE001
        ram_gb = None
        cpu_cores = os.cpu_count()
    return HardwareProfile(
        gpu_name=gpu_name,
        vram_total_gb=float(vram_total_gb),
        vram_free_gb=vram_free_gb,
        ram_gb=ram_gb,
        cpu_cores=cpu_cores,
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


def hardware_key(hardware: HardwareProfile) -> str:
    gpu_part = sanitize_filename_component(hardware.gpu_name or "cpu")
    vram_part = f"{hardware.vram_total_gb:.0f}GB"
    cpu_part = f"{hardware.cpu_cores or 0}c"
    return f"{gpu_part}_{vram_part}_{cpu_part}"


def profile_key(hardware: HardwareProfile, torch_version: str, model_id: str) -> str:
    hw_key = hardware_key(hardware)
    model_hash = hashlib.sha256(model_id.encode("utf-8")).hexdigest()[:8]
    torch_hash = hashlib.sha256(torch_version.encode("utf-8")).hexdigest()[:6]
    return f"{hw_key}_{torch_hash}_{model_hash}"


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


def base_preset_for_vram(vram_gb: float) -> dict[str, Any]:
    if vram_gb <= 6:
        return {
            "width": 448,
            "height": 448,
            "steps": 14,
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
            "steps": 16,
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
            "steps": 22,
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


def _candidate_grid() -> list[tuple[int, int]]:
    sizes = [768, 704, 640, 576, 512]
    steps = [28, 22, 18, 16, 14]
    candidates = []
    for size in sizes:
        for step in steps:
            candidates.append((size, step))
    return candidates


def autotune_sd_profile(
    pipeline,
    base_profile: dict[str, Any],
    quality_mode: str,
    logger: Logger | None = None,
    time_budget_s: float = 30.0,
    vram_safety: float = 0.75,
) -> dict[str, Any]:
    log = logger or _log_default
    prompt = "test anime still frame, high quality, sharp lineart"
    negative_prompt = "blurry, lowres, artifacts"
    generator = base_profile["generator"]
    best = base_profile
    start_time = time.time()

    try:
        pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=512,
            height=512,
            num_inference_steps=8,
            guidance_scale=base_profile["guidance"],
            generator=generator,
        )
    except Exception as exc:  # noqa: BLE001
        log(f"warmup failed: {exc}")

    candidates = _candidate_grid()
    if quality_mode == "fast":
        candidates = [(512, 14), (512, 16)]
    elif quality_mode == "auto":
        candidates = candidates[:8]

    for size, steps in candidates:
        if time.time() - start_time > time_budget_s:
            log("time budget exceeded; stopping probe")
            break
        try:
            pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=size,
                height=size,
                num_inference_steps=steps,
                guidance_scale=base_profile["guidance"],
                generator=generator,
            )
            vram_peak = 0.0
            try:
                import torch  # noqa: WPS433

                vram_peak = torch.cuda.max_memory_allocated() / (1024**3)
                torch.cuda.reset_peak_memory_stats()
            except Exception:  # noqa: BLE001
                vram_peak = 0.0
            if vram_peak and vram_peak > (base_profile["vram_total_gb"] * vram_safety):
                log(f"candidate {size}x{size} steps={steps} exceeds VRAM safety; stopping")
                break
            best = {**best, "width": size, "height": size, "steps": steps}
        except Exception as exc:  # noqa: BLE001
            if "out of memory" in str(exc).lower():
                log(f"OOM at {size}x{size} steps={steps}; stopping")
                try:
                    import torch  # noqa: WPS433

                    torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001
                    pass
                gc.collect()
                break
            log(f"probe failed: {exc}")
            break
    return best


def save_profile_from_info(profile_info: dict[str, Any]) -> None:
    if "profile_key" not in profile_info or "profile" not in profile_info:
        return
    save_cached_profile(profile_info["profile_key"], profile_info["profile"])


def get_quality_profile(
    pipeline,
    torch_module,
    model_id: str,
    quality_mode: str,
    logger: Logger | None = None,
) -> tuple[dict[str, Any], str]:
    log = logger or _log_default
    hardware = detect_hardware(torch_module)
    vram_total = hardware.vram_total_gb
    vram_free = hardware.vram_free_gb or vram_total
    vram_safety = float(os.getenv("MONEYOS_VRAM_SAFETY", "0.75"))
    time_budget = float(os.getenv("MONEYOS_QUALITY_TIME_BUDGET_SECONDS", "30"))

    base_profile = base_preset_for_vram(vram_free or vram_total)
    base_profile.update(
        {
            "vram_total_gb": vram_total,
            "vram_free_gb": vram_free,
            "generator": torch_module.Generator(
                device="cuda" if torch_module.cuda.is_available() else "cpu"
            ).manual_seed(1234),
        }
    )
    key = profile_key(hardware, torch_module.__version__, model_id)
    source = "defaults"
    if quality_mode != "manual" and os.getenv("MONEYOS_QUALITY_FORCE_PROBE", "0") != "1":
        cached = load_cached_profile(key)
        if cached:
            base_profile.update(cached)
            source = "cached"
            log(f"cache hit: {key}")
            return {**base_profile, "profile_key": key}, source
    if quality_mode != "manual":
        log(f"probing profile (mode={quality_mode})")
        tuned = autotune_sd_profile(
            pipeline,
            base_profile,
            quality_mode,
            logger=log,
            time_budget_s=time_budget,
            vram_safety=vram_safety,
        )
        profile_payload = {
            "width": tuned["width"],
            "height": tuned["height"],
            "steps": tuned["steps"],
            "guidance": tuned["guidance"],
            "fp16": tuned["fp16"],
            "cpu_offload": tuned["cpu_offload"],
            "attention_slicing": tuned["attention_slicing"],
            "vae_slicing": tuned["vae_slicing"],
            "xformers": tuned["xformers"],
            "vram_safety": vram_safety,
            "time_budget_s": time_budget,
            "timestamp": time.time(),
        }
        save_cached_profile(key, profile_payload)
        source = "probed"
        return {**base_profile, **profile_payload, "profile_key": key}, source
    return {**base_profile, "profile_key": key}, "manual"
