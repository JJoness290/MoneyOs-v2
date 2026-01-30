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


_PIPELINE = None
_PIPELINE_MODEL = None
_PIPELINE_LOCK = threading.Lock()
_LOGGED_MEMORY = False

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


def _detect_vram_gb(torch_module) -> tuple[float, float]:
    if not torch_module.cuda.is_available():
        return 0.0, 0.0
    props = torch_module.cuda.get_device_properties(0)
    total_gb = props.total_memory / (1024**3)
    free_gb = 0.0
    try:
        free_gb = torch_module.cuda.mem_get_info()[0] / (1024**3)
    except Exception:  # noqa: BLE001
        free_gb = 0.0
    return total_gb, free_gb


def _model_family(model_name: str) -> str:
    if model_name == "sdxl":
        return "sdxl"
    return "sd15"


def pick_defaults(vram_gb: float, model_family: str) -> dict[str, Any]:
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
        }
    if vram_gb <= 6:
        return {
            "width": 384,
            "height": 384,
            "steps": 12,
            "guidance": 6.0,
            "fp16": True,
            "cpu_offload": True,
            "attention_slicing": True,
            "vae_slicing": True,
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
        }
    if vram_gb <= 12:
        return {
            "width": 640,
            "height": 640,
            "steps": 18,
            "guidance": 6.5,
            "fp16": True,
            "cpu_offload": False,
            "attention_slicing": True,
            "vae_slicing": True,
        }
    return {
        "width": 768,
        "height": 768,
        "steps": 24,
        "guidance": 6.5,
        "fp16": True,
        "cpu_offload": False,
        "attention_slicing": False,
        "vae_slicing": False,
    }


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value not in {"0", "false", "False"}


def _clamp_dim(value: int) -> int:
    value = max(256, value)
    return int(round(value / 8) * 8)


def _resolve_settings(torch_module, settings: SDSettings) -> SDSettings:
    total_gb, free_gb = _detect_vram_gb(torch_module)
    vram_gb = free_gb or total_gb
    family = _model_family(settings.model)
    defaults = pick_defaults(vram_gb, family)
    width = _clamp_dim(int(os.getenv("MONEYOS_SD_WIDTH", defaults["width"])))
    height = _clamp_dim(int(os.getenv("MONEYOS_SD_HEIGHT", defaults["height"])))
    steps = int(os.getenv("MONEYOS_SD_STEPS", defaults["steps"]))
    guidance = float(os.getenv("MONEYOS_SD_GUIDANCE", defaults["guidance"]))
    fp16 = _env_bool("MONEYOS_SD_FP16", defaults["fp16"])
    cpu_offload = _env_bool("MONEYOS_SD_CPU_OFFLOAD", defaults["cpu_offload"])
    attn_slicing = _env_bool("MONEYOS_SD_ATTENTION_SLICING", defaults["attention_slicing"])
    vae_slicing = _env_bool("MONEYOS_SD_VAE_SLICING", defaults["vae_slicing"])
    print(
        "[ANIME] vram total="
        f"{total_gb:.2f}GB free={free_gb:.2f}GB using={vram_gb:.2f}GB "
        f"defaults={defaults}"
    )
    print(
        "[ANIME] effective settings="
        f"{width}x{height} steps={steps} guidance={guidance} "
        f"fp16={fp16} cpu_offload={cpu_offload} attn_slicing={attn_slicing} vae_slicing={vae_slicing}"
    )
    return SDSettings(
        model=settings.model,
        steps=steps,
        guidance=guidance,
        seed=settings.seed,
        width=width,
        height=height,
        fp16=fp16,
        cpu_offload=cpu_offload,
        attention_slicing=attn_slicing,
        vae_slicing=vae_slicing,
    )


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
    effective_settings = _resolve_settings(torch, settings)
    cache_key = _hash_prompt(prompt, negative_prompt, effective_settings)
    cached_path = cache_dir / f"{cache_key}.png"
    if AI_IMAGE_CACHE and cached_path.exists():
        return cached_path

    global _PIPELINE, _PIPELINE_MODEL
    with _PIPELINE_LOCK:
        if _PIPELINE is None or _PIPELINE_MODEL != effective_settings.model:
            pipeline, device = _prepare_pipeline(effective_settings)
            _PIPELINE = pipeline
            _PIPELINE_MODEL = effective_settings.model
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline = _PIPELINE
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
