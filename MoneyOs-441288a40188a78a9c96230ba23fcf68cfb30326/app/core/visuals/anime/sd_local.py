from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import importlib.util
import os
from pathlib import Path
import threading
from typing import Any

from app.config import (
    AI_IMAGE_CACHE,
    ANIME_LORA_PATHS,
    ANIME_LORA_WEIGHTS,
    SD_GUIDANCE,
    SD_MAX_BATCH_SIZE,
    SD_MODEL,
    SD_MODEL_ID,
    SD_MODEL_LOCAL_PATH,
    SD_MODEL_SOURCE,
    SD_PROFILE,
    SD_SEED,
    SD_STEPS,
)
from app.core.quality_profiles import (
    detect_hardware,
    base_preset_for_vram,
    get_quality_profile,
    sanitize_filename_component,
    save_profile_from_info,
)


_PIPELINE = None
_PIPELINE_MODEL = None
_PIPELINE_LOCK = threading.Lock()
_LOGGED_MEMORY = False
_PROFILE_INFO: dict[str, Any] | None = None

_MODEL_PRESETS = {
    "sd15_anime": "runwayml/stable-diffusion-v1-5",
    "sdxl_anime": "stabilityai/stable-diffusion-xl-base-1.0",
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
    batch_size: int


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
    env_override = os.getenv("MONEYOS_SD_MODEL_ID")
    if env_override:
        return env_override
    if model_name == "sd15_anime":
        return SD_MODEL_ID
    return _MODEL_PRESETS.get(model_name, _MODEL_PRESETS["sd15_anime"])


def _model_family(model_name: str) -> str:
    if model_name == "sdxl_anime":
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
    if mode not in {"auto", "max", "low", "balanced", "high", "manual"}:
        return "auto"
    return mode


def _select_model_name(hardware, quality_mode: str) -> str:
    explicit = os.getenv("MONEYOS_SD_MODEL")
    if explicit:
        return SD_MODEL
    if hardware.vram_total_gb <= 8:
        return "sd15_anime"
    if hardware.vram_total_gb >= 12 and quality_mode in {"high", "max"}:
        return "sdxl_anime"
    return "sd15_anime"


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
    if _quality_mode() == "manual":
        return settings
    if SD_PROFILE == "max":
        return settings
    hardware = detect_hardware(torch_module)
    vram_gb = hardware.vram_free_gb or hardware.vram_total_gb
    base_profile = base_preset_for_vram(vram_gb)
    fp16 = _env_bool("MONEYOS_SD_FP16", base_profile["fp16"])
    cpu_offload = _env_bool("MONEYOS_SD_CPU_OFFLOAD", base_profile["cpu_offload"])
    attn_slicing = _env_bool("MONEYOS_SD_ATTENTION_SLICING", base_profile["attention_slicing"])
    vae_slicing = _env_bool("MONEYOS_SD_VAE_SLICING", base_profile["vae_slicing"])
    return SDSettings(
        model=settings.model,
        steps=settings.steps,
        guidance=settings.guidance,
        seed=settings.seed,
        width=settings.width,
        height=settings.height,
        fp16=fp16,
        cpu_offload=cpu_offload,
        attention_slicing=attn_slicing,
        vae_slicing=vae_slicing,
        batch_size=settings.batch_size,
    )


def _sd_profile_settings(base: SDSettings) -> SDSettings:
    profile = SD_PROFILE
    if profile == "max":
        return SDSettings(
            model=base.model,
            steps=40,
            guidance=6.5,
            seed=base.seed,
            width=1024,
            height=1024,
            fp16=True,
            cpu_offload=False,
            attention_slicing=False,
            vae_slicing=False,
            batch_size=max(1, SD_MAX_BATCH_SIZE),
        )
    return SDSettings(
        model=base.model,
        steps=max(base.steps, 28),
        guidance=6.5,
        seed=base.seed,
        width=max(base.width, 768),
        height=max(base.height, 768),
        fp16=True,
        cpu_offload=base.cpu_offload,
        attention_slicing=True,
        vae_slicing=True,
        batch_size=1,
    )


def _log_lora_status() -> None:
    if ANIME_LORA_PATHS:
        print(
            "[ANIME] LoRA paths provided but LoRA loading is not implemented yet. "
            "Set MONEYOS_ANIME_LORA_PATHS empty to silence this warning."
        )


def _resolve_profile(torch_module, settings: SDSettings, pipeline) -> tuple[SDSettings, dict[str, Any]]:
    mode = _quality_mode()
    hardware = detect_hardware(torch_module)
    if mode == "manual":
        profile = {
            "width": settings.width,
            "height": settings.height,
            "steps": settings.steps,
            "guidance": settings.guidance,
            "fp16": settings.fp16,
            "cpu_offload": settings.cpu_offload,
            "attention_slicing": settings.attention_slicing,
            "vae_slicing": settings.vae_slicing,
            "batch_size": settings.batch_size,
            "xformers": os.getenv("MONEYOS_SD_XFORMERS", "0") == "1",
        }
        final_profile = _merge_env_overrides(profile)
        source = "manual"
    else:
        model_id = _resolve_model_id(settings.model)
        profile, source = get_quality_profile(
            pipeline,
            torch_module,
            model_id,
            mode,
            logger=lambda msg: print(f"[ANIME] {msg}"),
        )
        final_profile = _merge_env_overrides(profile)
    print(
        "[ANIME] hardware="
        f"{sanitize_filename_component(hardware.gpu_name or 'cpu')} "
        f"vram_total={hardware.vram_total_gb:.2f}GB "
        f"vram_free={(hardware.vram_free_gb or 0.0):.2f}GB"
    )
    print(f"[ANIME] quality_mode={mode} profile_source={source}")
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
        batch_size=int(final_profile.get("batch_size", settings.batch_size)),
    )
    global _PROFILE_INFO
    _PROFILE_INFO = {
        "quality_mode": mode,
        "profile_source": source,
        "profile_key": final_profile.get("profile_key"),
        "profile": final_profile,
    }
    return settings_out, _PROFILE_INFO


def _prepare_pipeline(settings: SDSettings):
    torch = importlib.import_module("torch")
    diffusers = importlib.import_module("diffusers")
    use_gpu = os.getenv("MONEYOS_USE_GPU", "0") == "1"
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model_id = _resolve_model_id(settings.model)
    dtype = torch.float16 if device == "cuda" and settings.fp16 else torch.float32
    if SD_MODEL_SOURCE == "local_ckpt":
        if not SD_MODEL_LOCAL_PATH.exists():
            raise RuntimeError(
                "Local SD model not found at "
                f"{SD_MODEL_LOCAL_PATH}. Set MONEYOS_SD_MODEL_LOCAL_PATH "
                "or use MONEYOS_SD_MODEL_SOURCE=diffusers_hf."
            )
        if hasattr(diffusers.DiffusionPipeline, "from_single_file"):
            pipeline = diffusers.DiffusionPipeline.from_single_file(
                str(SD_MODEL_LOCAL_PATH),
                torch_dtype=dtype,
            )
        else:
            raise RuntimeError("Diffusers does not support from_single_file for local_ckpt models.")
    else:
        pipeline = diffusers.DiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    if settings.cpu_offload and device == "cuda":
        pipeline.enable_model_cpu_offload()
    else:
        pipeline = pipeline.to(device)
    _log_lora_status()
    if device == "cuda":
        try:
            if hasattr(torch.backends.cuda, "enable_flash_sdp"):
                torch.backends.cuda.enable_flash_sdp(True)
            if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
                torch.backends.cuda.enable_mem_efficient_sdp(True)
            if hasattr(torch.backends.cuda, "enable_math_sdp"):
                torch.backends.cuda.enable_math_sdp(True)
        except Exception as exc:  # noqa: BLE001
            print(f"[ANIME] SDP attention config failed ({exc})")
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
        batch_size=1,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch = importlib.import_module("torch")

    global _PIPELINE, _PIPELINE_MODEL
    hardware = detect_hardware(torch)
    selected_model = _select_model_name(hardware, _quality_mode())
    if selected_model != settings.model:
        settings = SDSettings(
            model=selected_model,
            steps=settings.steps,
            guidance=settings.guidance,
            seed=settings.seed,
            width=settings.width,
            height=settings.height,
            fp16=settings.fp16,
            cpu_offload=settings.cpu_offload,
            attention_slicing=settings.attention_slicing,
            vae_slicing=settings.vae_slicing,
            batch_size=settings.batch_size,
        )
    profile_settings = _sd_profile_settings(settings)
    pipeline_settings = _pipeline_settings(torch, profile_settings)
    with _PIPELINE_LOCK:
        if _PIPELINE is None or _PIPELINE_MODEL != pipeline_settings.model:
            pipeline, device = _prepare_pipeline(pipeline_settings)
            _PIPELINE = pipeline
            _PIPELINE_MODEL = pipeline_settings.model
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline = _PIPELINE
    effective_settings, profile_info = _resolve_profile(torch, pipeline_settings, pipeline)
    cache_key = _hash_prompt(prompt, negative_prompt, effective_settings)
    cached_path = cache_dir / f"{cache_key}.png"
    if AI_IMAGE_CACHE and cached_path.exists():
        return cached_path
    generator = torch.Generator(device=device)
    if effective_settings.seed != 0:
        generator = generator.manual_seed(effective_settings.seed)
    else:
        generator = generator.manual_seed(1234)

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
    print(
        "[ANIME] sd_profile="
        f"{SD_PROFILE} model_source={SD_MODEL_SOURCE} "
        f"settings={effective_settings.width}x{effective_settings.height} "
        f"steps={effective_settings.steps} cfg={effective_settings.guidance} "
        f"batch_size={effective_settings.batch_size} fp16={effective_settings.fp16} "
        f"attn_slicing={effective_settings.attention_slicing} "
        f"vae_slicing={effective_settings.vae_slicing}"
    )

    def _run_inference(width_val: int, height_val: int, steps_val: int, batch_size: int):
        if device == "cuda":
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:  # noqa: BLE001
                pass
        result = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps_val,
            guidance_scale=effective_settings.guidance,
            width=width_val,
            height=height_val,
            num_images_per_prompt=batch_size,
            generator=generator,
        )
        if device == "cuda":
            try:
                peak = torch.cuda.max_memory_allocated() / (1024**3)
                print(f"[ANIME] device={device} dtype={pipeline.unet.dtype} peak_vram={peak:.2f}GB")
            except Exception:  # noqa: BLE001
                pass
        return result

    def _degrade_profile(settings_obj: SDSettings) -> SDSettings:
        size_ladder = [settings_obj.width, 704, 640, 576, 512]
        step_ladder = [settings_obj.steps, 22, 18, 16, 14]
        size_ladder = [value for value in size_ladder if value >= 512]
        step_ladder = [value for value in step_ladder if value >= 14]
        new_width = size_ladder[1] if len(size_ladder) > 1 else settings_obj.width
        new_steps = step_ladder[1] if len(step_ladder) > 1 else settings_obj.steps
        return SDSettings(
            model=settings_obj.model,
            steps=new_steps,
            guidance=settings_obj.guidance,
            seed=settings_obj.seed,
            width=new_width,
            height=new_width,
            fp16=True,
            cpu_offload=True,
            attention_slicing=True,
            vae_slicing=True,
            batch_size=1,
        )

    attempt_settings = effective_settings
    max_retry = 2 if SD_PROFILE == "max" else 1
    downgraded = False
    for attempt in range(3):
        try:
            if device == "cuda" and attempt_settings.fp16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    result = _run_inference(
                        attempt_settings.width,
                        attempt_settings.height,
                        attempt_settings.steps,
                        attempt_settings.batch_size,
                    )
            else:
                result = _run_inference(
                    attempt_settings.width,
                    attempt_settings.height,
                    attempt_settings.steps,
                    attempt_settings.batch_size,
                )
            break
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and device == "cuda":
                torch.cuda.empty_cache()
                print(
                    "[ANIME] CUDA OOM, degrading profile "
                    f"({attempt_settings.width}x{attempt_settings.height} -> lower)"
                )
                if SD_PROFILE == "max" and attempt_settings.batch_size > 1:
                    attempt_settings = SDSettings(
                        model=attempt_settings.model,
                        steps=attempt_settings.steps,
                        guidance=attempt_settings.guidance,
                        seed=attempt_settings.seed,
                        width=attempt_settings.width,
                        height=attempt_settings.height,
                        fp16=attempt_settings.fp16,
                        cpu_offload=attempt_settings.cpu_offload,
                        attention_slicing=attempt_settings.attention_slicing,
                        vae_slicing=attempt_settings.vae_slicing,
                        batch_size=1,
                    )
                    downgraded = True
                elif SD_PROFILE == "max" and attempt_settings.width >= 1024:
                    attempt_settings = SDSettings(
                        model=attempt_settings.model,
                        steps=attempt_settings.steps,
                        guidance=attempt_settings.guidance,
                        seed=attempt_settings.seed,
                        width=896,
                        height=896,
                        fp16=attempt_settings.fp16,
                        cpu_offload=attempt_settings.cpu_offload,
                        attention_slicing=attempt_settings.attention_slicing,
                        vae_slicing=attempt_settings.vae_slicing,
                        batch_size=attempt_settings.batch_size,
                    )
                    downgraded = True
                else:
                    attempt_settings = _degrade_profile(attempt_settings)
                if attempt == 1:
                    profile_info = profile_info or {}
                    profile = profile_info.get("profile", {})
                    if profile:
                        profile.update(
                            {
                                "width": attempt_settings.width,
                                "height": attempt_settings.height,
                                "steps": attempt_settings.steps,
                                "fp16": attempt_settings.fp16,
                                "cpu_offload": attempt_settings.cpu_offload,
                                "attention_slicing": attempt_settings.attention_slicing,
                                "vae_slicing": attempt_settings.vae_slicing,
                                "batch_size": attempt_settings.batch_size,
                            }
                        )
                        save_profile_from_info(profile_info)
                continue
            raise
    if downgraded:
        print("[ANIME] max profile downgraded after OOM (batch size/resolution reduced)")
    image = result.images[0]
    image.save(output_path)
    if AI_IMAGE_CACHE:
        cached_path.write_bytes(output_path.read_bytes())
        return cached_path
    return output_path
