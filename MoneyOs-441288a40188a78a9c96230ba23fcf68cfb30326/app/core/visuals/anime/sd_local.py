from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import importlib.util
from pathlib import Path
import threading

from app.config import AI_IMAGE_CACHE, SD_GUIDANCE, SD_MODEL, SD_MODEL_ID, SD_SEED, SD_STEPS


_PIPELINE = None
_PIPELINE_MODEL = None
_PIPELINE_LOCK = threading.Lock()

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


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def is_sd_available() -> bool:
    return _has_module("torch") and _has_module("diffusers")


def _hash_prompt(prompt: str, negative_prompt: str, settings: SDSettings, width: int, height: int) -> str:
    raw = (
        f"{prompt}|{negative_prompt}|{settings.model}|{settings.steps}|{settings.guidance}|"
        f"{settings.seed}|{width}x{height}|{SD_MODEL_ID}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _resolve_model_id(model_name: str) -> str:
    if SD_MODEL_ID:
        return SD_MODEL_ID
    return _MODEL_PRESETS.get(model_name, _MODEL_PRESETS["sd15_anime"])


def _prepare_pipeline(settings: SDSettings):
    torch = importlib.import_module("torch")
    diffusers = importlib.import_module("diffusers")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model_id = _resolve_model_id(settings.model)
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipeline = diffusers.DiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    if settings.model == "sdxl" and device == "cuda":
        pipeline.enable_model_cpu_offload()
    else:
        pipeline = pipeline.to(device)
    if hasattr(pipeline, "enable_xformers_memory_efficient_attention"):
        try:
            pipeline.enable_xformers_memory_efficient_attention()
        except Exception as exc:  # noqa: BLE001
            print(f"[ANIME] xformers unavailable ({exc}); using default attention")
    if hasattr(pipeline, "enable_attention_slicing"):
        pipeline.enable_attention_slicing()
    if hasattr(pipeline, "enable_vae_slicing"):
        pipeline.enable_vae_slicing()
    if hasattr(pipeline, "enable_vae_tiling"):
        pipeline.enable_vae_tiling()
    return pipeline, device


def generate_image(
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    output_path: Path,
    cache_dir: Path,
    settings: SDSettings | None = None,
) -> Path:
    settings = settings or SDSettings(model=SD_MODEL, steps=SD_STEPS, guidance=SD_GUIDANCE, seed=SD_SEED)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = _hash_prompt(prompt, negative_prompt, settings, width, height)
    cached_path = cache_dir / f"{cache_key}.png"
    if AI_IMAGE_CACHE and cached_path.exists():
        return cached_path

    torch = importlib.import_module("torch")
    global _PIPELINE, _PIPELINE_MODEL
    with _PIPELINE_LOCK:
        if _PIPELINE is None or _PIPELINE_MODEL != settings.model:
            pipeline, device = _prepare_pipeline(settings)
            _PIPELINE = pipeline
            _PIPELINE_MODEL = settings.model
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline = _PIPELINE
    generator = torch.Generator(device=device)
    if settings.seed != 0:
        generator = generator.manual_seed(settings.seed)

    if device == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            result = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=settings.steps,
                guidance_scale=settings.guidance,
                width=width,
                height=height,
                generator=generator,
            )
    else:
        result = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=settings.steps,
            guidance_scale=settings.guidance,
            width=width,
            height=height,
            generator=generator,
        )
    image = result.images[0]
    image.save(output_path)
    if AI_IMAGE_CACHE:
        cached_path.write_bytes(output_path.read_bytes())
        return cached_path
    return output_path
