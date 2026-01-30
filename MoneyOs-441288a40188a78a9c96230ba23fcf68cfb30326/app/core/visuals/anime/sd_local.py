from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import importlib.util
import os
from pathlib import Path
import threading
from typing import Any

from app.config import AI_IMAGE_CACHE, SD_GUIDANCE, SD_MODEL, SD_MODEL_ID, SD_SEED, SD_STEPS
from app.core.quality_autotune import (
    autotune_sd_profile,
    base_preset_for_vram,
    detect_hardware,
    load_cached_profile,
    profile_key,
    sanitize_filename_component,
    save_cached_profile,
)


_PIPELINE = None
_PIPELINE_MODEL = None
_PIPELINE_LOCK = threading.Lock()
_LOGGED_MEMORY = False
_PROFILE_INFO: dict[str, Any] | None = None

_MODEL_PRESETS = {
    "sd15_anime": "Linaqruf/anything-v4.0",
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
}


@dataclass(frozen=True)
class SDSettings:
    model: str
    steps: int
    guidance: float
    seed: int
    width: int
    height: int
    fp16: bool
    cpu_offload: bool
    attention_slicing: bool
    vae_slicing: bool


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def is_sd_available() -> bool:
    return _has_module("torch") and _has_module("diffusers")


def get_quality_profile_info() -> dict[str, Any] | None:
    return _PROFILE_INFO


def _hash_prompt(prompt: str, negative_prompt: str, settings: SDSettings) -> str:
    raw = (
        f"{prompt}|{negative_prompt}|{settings.model}|{settings.steps}|{settings.guidance}|"
        f"{settings.seed}|{settings.width}x{settings.height}|{settings.fp16}|"
        f"{settings.cpu_offload}|{settings.attention_slicing}|{settings.vae_slicing}|{SD_MODEL_ID}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _resolve_model_id(model_name: str) -> str:
    if SD_MODEL_ID:
        return SD_MODEL_ID
    return _MODEL_PRESETS.get(model_name, _MODEL_PRESETS["sd15_anime"])


def _model_family(model_name: str) -> str:
    if model_name == "sdxl":
        return "sdxl"
    return "sd15"


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value not in {"0", "false", "False"}


def _clamp_dim(value: int) -> int:
    value = max(256, value)
    return int(round(value / 8) * 8)


def _quality_mode() -> str:
    mode = os.getenv("MONEYOS_QUALITY", "auto").strip().lower()
    if mode not in {"auto", "max", "fast", "manual"}:
        return "auto"
    return mode


def _merge_env_overrides(profile: dict[str, Any]) -> dict[str, Any]:
    overrides = dict(profile)
    width_env = os.getenv("MONEYOS_SD_WIDTH")
    height_env = os.getenv("MONEYOS_SD_HEIGHT")
    steps_env = os.getenv("MONEYOS_SD_STEPS")
    guidance_env = os.getenv("MONEYOS_SD_GUIDANCE")
    if width_env:
        overrides["width"] = _clamp_dim(int(width_env))
    if height_env:
        overrides["height"] = _clamp_dim(int(height_env))
    if steps_env:
        overrides["steps"] = int(steps_env)
    if guidance_env:
        overrides["guidance"] = float(guidance_env)
    overrides["fp16"] = _env_bool("MONEYOS_SD_FP16", overrides["fp16"])
    overrides["cpu_offload"] = _env_bool("MONEYOS_SD_CPU_OFFLOAD", overrides["cpu_offload"])
    overrides["attention_slicing"] = _env_bool("MONEYOS_SD_ATTENTION_SLICING", overrides["attention_slicing"])
    overrides["vae_slicing"] = _env_bool("MONEYOS_SD_VAE_SLICING", overrides["vae_slicing"])
    overrides["xformers"] = os.getenv("MONEYOS_SD_XFORMERS", "0") == "1"
    return overrides


def _pipeline_settings(torch_module, settings: SDSettings) -> SDSettings:
    mode = _quality_mode()
    if mode == "manual":
        return settings
    hardware = detect_hardware(torch_module)
    vram_gb = hardware.vram_free_gb or hardware.vram_total_gb
    family = _model_family(settings.model)
    base_profile = base_preset_for_vram(vram_gb, family)
    fp16 = _env_bool("MONEYOS_SD_FP16", base_profile["fp16"])
    cpu_offload = _env_bool("MONEYOS_SD_CPU_OFFLOAD", base_profile["cpu_offload"])
    attn_slicing = _env_bool("MONEYOS_SD_ATTENTION_SLICING", base_profile["attention_slicing"])
    vae_slicing = _env_bool("MONEYOS_SD_VAE_SLICING", base_profile["vae_slicing"])
    return SDSettings(
        model=settings.model,
        steps=base_profile["steps"],
        guidance=base_profile["guidance"],
        seed=settings.seed,
        width=base_profile["width"],
        height=base_profile["height"],
        fp16=fp16,
        cpu_offload=cpu_offload,
        attention_slicing=attn_slicing,
        vae_slicing=vae_slicing,
    )


def _resolve_profile(torch_module, settings: SDSettings, pipeline) -> tuple[SDSettings, dict[str, Any]]:
    mode = _quality_mode()
    hardware = detect_hardware(torch_module)
    vram_gb = hardware.vram_free_gb or hardware.vram_total_gb
    family = _model_family(settings.model)
    if mode == "manual":
        base_profile = {
            "width": _clamp_dim(int(os.getenv("MONEYOS_SD_WIDTH", settings.width))),
            "height": _clamp_dim(int(os.getenv("MONEYOS_SD_HEIGHT", settings.height))),
            "steps": int(os.getenv("MONEYOS_SD_STEPS", settings.steps)),
            "guidance": float(os.getenv("MONEYOS_SD_GUIDANCE", settings.guidance)),
            "fp16": _env_bool("MONEYOS_SD_FP16", settings.fp16),
            "cpu_offload": _env_bool("MONEYOS_SD_CPU_OFFLOAD", settings.cpu_offload),
            "attention_slicing": _env_bool("MONEYOS_SD_ATTENTION_SLICING", settings.attention_slicing),
            "vae_slicing": _env_bool("MONEYOS_SD_VAE_SLICING", settings.vae_slicing),
            "xformers": os.getenv("MONEYOS_SD_XFORMERS", "0") == "1",
        }
    else:
        base_profile = base_preset_for_vram(vram_gb, family)
    base_profile.update(
        {
            "model_family": family,
            "vram_gb": vram_gb,
            "guidance": base_profile["guidance"],
            "generator": torch_module.Generator(device="cuda" if torch_module.cuda.is_available() else "cpu")
            .manual_seed(1234),
            "pipeline": pipeline,
        }
    )
    model_id = _resolve_model_id(settings.model)
    key = profile_key(hardware, torch_module.__version__, model_id)
    cached = None
    source = "probed"
    if mode != "manual" and os.getenv("MONEYOS_QUALITY_FORCE_PROBE", "0") != "1":
        cached = load_cached_profile(key)
        if cached:
            source = "cache_hit"
            base_profile.update(cached)
    if mode != "manual" and not cached:
        debug = os.getenv("MONEYOS_DEBUG_QUALITY", "0") == "1"
        base_profile = autotune_sd_profile(
            base_profile["pipeline"],
            base_profile,
            mode,
            logger=lambda msg: print(f"[ANIME] {msg}"),
            debug=debug,
        )
        save_cached_profile(key, {k: base_profile[k] for k in ["width", "height", "steps", "guidance"]})
    if mode == "manual":
        source = "manual"
    final_profile = _merge_env_overrides(base_profile) if mode != "manual" else base_profile
    print(
        "[ANIME] hardware="
        f"{sanitize_filename_component(hardware.gpu_name or 'cpu')} "
        f"vram_total={hardware.vram_total_gb:.2f}GB "
        f"vram_free={(hardware.vram_free_gb or 0.0):.2f}GB"
    )
    print(f"[ANIME] quality_mode={mode} profile_source={source} profile_key={key}")
    print(
        "[ANIME] final profile "
        f"{final_profile['width']}x{final_profile['height']} steps={final_profile['steps']} "
        f"guidance={final_profile['guidance']} fp16={final_profile['fp16']} "
        f"cpu_offload={final_profile['cpu_offload']} attn_slicing={final_profile['attention_slicing']} "
        f"vae_slicing={final_profile['vae_slicing']} xformers={final_profile['xformers']}"
    )
    settings_out = SDSettings(
        model=settings.model,
        steps=int(final_profile["steps"]),
        guidance=float(final_profile["guidance"]),
        seed=settings.seed,
        width=_clamp_dim(int(final_profile["width"])),
        height=_clamp_dim(int(final_profile["height"])),
        fp16=bool(final_profile["fp16"]),
        cpu_offload=bool(final_profile["cpu_offload"]),
        attention_slicing=bool(final_profile["attention_slicing"]),
        vae_slicing=bool(final_profile["vae_slicing"]),
    )
    global _PROFILE_INFO
    _PROFILE_INFO = {
        "quality_mode": mode,
        "profile_source": source,
        "profile_key": key,
        "profile": final_profile,
    }
    return settings_out, _PROFILE_INFO


def _prepare_pipeline(settings: SDSettings):
    torch = importlib.import_module("torch")
    diffusers = importlib.import_module("diffusers")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model_id = _resolve_model_id(settings.model)
    dtype = torch.float16 if device == "cuda" and settings.fp16 else torch.float32
    pipeline = diffusers.DiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    if settings.cpu_offload and device == "cuda":
        pipeline.enable_model_cpu_offload()
    else:
        pipeline = pipeline.to(device)
    if os.getenv("MONEYOS_SD_XFORMERS", "0") == "1" and hasattr(
        pipeline, "enable_xformers_memory_efficient_attention"
    ):
        try:
            pipeline.enable_xformers_memory_efficient_attention()
        except Exception as exc:  # noqa: BLE001
            print(f"[ANIME] xformers unavailable ({exc}); using default attention")
    if settings.attention_slicing and hasattr(pipeline, "enable_attention_slicing"):
        pipeline.enable_attention_slicing()
    if settings.vae_slicing and hasattr(pipeline, "enable_vae_slicing"):
        pipeline.enable_vae_slicing()
    if settings.vae_slicing and hasattr(pipeline, "enable_vae_tiling"):
        pipeline.enable_vae_tiling()
    return pipeline, device


def generate_image(
    prompt: str,
    negative_prompt: str,
    output_path: Path,
    cache_dir: Path,
    settings: SDSettings | None = None,
) -> Path:
    settings = settings or SDSettings(
        model=SD_MODEL,
        steps=SD_STEPS,
        guidance=SD_GUIDANCE,
        seed=SD_SEED,
        width=512,
        height=512,
        fp16=True,
        cpu_offload=False,
        attention_slicing=True,
        vae_slicing=True,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch = importlib.import_module("torch")

    global _PIPELINE, _PIPELINE_MODEL
    pipeline_settings = _pipeline_settings(torch, settings)
    with _PIPELINE_LOCK:
        if _PIPELINE is None or _PIPELINE_MODEL != pipeline_settings.model:
            pipeline, device = _prepare_pipeline(pipeline_settings)
            _PIPELINE = pipeline
            _PIPELINE_MODEL = pipeline_settings.model
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline = _PIPELINE
    effective_settings, _ = _resolve_profile(torch, settings, pipeline)
    cache_key = _hash_prompt(prompt, negative_prompt, effective_settings)
    cached_path = cache_dir / f"{cache_key}.png"
    if AI_IMAGE_CACHE and cached_path.exists():
        return cached_path
    generator = torch.Generator(device=device)
    if effective_settings.seed != 0:
        generator = generator.manual_seed(effective_settings.seed)

    global _LOGGED_MEMORY
    if not _LOGGED_MEMORY:
        try:
            mem_alloc = torch.cuda.memory_allocated() / (1024**3)
            mem_reserved = torch.cuda.memory_reserved() / (1024**3)
            print(
                "[ANIME] cuda memory allocated="
                f"{mem_alloc:.2f}GB reserved={mem_reserved:.2f}GB device={device}"
            )
            try:
                unet = getattr(pipeline, "unet", None)
                if unet is not None:
                    print(f"[ANIME] unet dtype={unet.dtype} device={unet.device}")
            except Exception:  # noqa: BLE001
                pass
        except Exception:  # noqa: BLE001
            pass
        _LOGGED_MEMORY = True

    def _run_inference(width_val: int, height_val: int, steps_val: int):
        result = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps_val,
            guidance_scale=effective_settings.guidance,
            width=width_val,
            height=height_val,
            generator=generator,
        )
        return result

    try:
        if device == "cuda" and effective_settings.fp16:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                result = _run_inference(effective_settings.width, effective_settings.height, effective_settings.steps)
        else:
            result = _run_inference(effective_settings.width, effective_settings.height, effective_settings.steps)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower() and device == "cuda":
            torch.cuda.empty_cache()
            down_width = _clamp_dim(max(256, int(effective_settings.width * 0.75)))
            down_height = _clamp_dim(max(256, int(effective_settings.height * 0.75)))
            down_steps = max(8, int(effective_settings.steps * 0.75))
            print(
                "[ANIME] CUDA OOM retry with "
                f"{down_width}x{down_height} steps={down_steps}"
            )
            if effective_settings.fp16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    result = _run_inference(down_width, down_height, down_steps)
            else:
                result = _run_inference(down_width, down_height, down_steps)
        else:
            raise
    image = result.images[0]
    image.save(output_path)
    if AI_IMAGE_CACHE:
        cached_path.write_bytes(output_path.read_bytes())
        return cached_path
    return output_path
