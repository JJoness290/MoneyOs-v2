from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import importlib.util
from pathlib import Path

from app.config import AI_IMAGE_CACHE, SD_GUIDANCE, SD_MODEL, SD_SEED, SD_STEPS


_PIPELINE = None
_PIPELINE_MODEL = None


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
    raw = f"{prompt}|{negative_prompt}|{settings.model}|{settings.steps}|{settings.guidance}|{settings.seed}|{width}x{height}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


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
    diffusers = importlib.import_module("diffusers")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    global _PIPELINE, _PIPELINE_MODEL
    if _PIPELINE is None or _PIPELINE_MODEL != settings.model:
        pipeline = diffusers.DiffusionPipeline.from_pretrained(settings.model, torch_dtype=dtype)
        pipeline = pipeline.to(device)
        if hasattr(pipeline, "enable_xformers_memory_efficient_attention"):
            pipeline.enable_xformers_memory_efficient_attention()
        if hasattr(pipeline, "enable_vae_slicing"):
            pipeline.enable_vae_slicing()
        if hasattr(pipeline, "enable_vae_tiling"):
            pipeline.enable_vae_tiling()
        _PIPELINE = pipeline
        _PIPELINE_MODEL = settings.model
    pipeline = _PIPELINE
    generator = torch.Generator(device=device)
    if settings.seed != 0:
        generator = generator.manual_seed(settings.seed)

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
