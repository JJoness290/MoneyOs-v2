from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Optional

from app.config import OUTPUT_DIR


@dataclass
class GenerationResult:
    path: Path
    size: int
    device: str


_PIPELINE = None
_PIPELINE_DEVICE = None


def cache_key_for_prompt(
    prompt: str,
    negative_prompt: str,
    *,
    size: int,
    seed: int,
    style_preset: str,
    model_id: str,
) -> str:
    payload = f"{prompt}|{negative_prompt}|{size}|{seed}|{style_preset}|{model_id}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _load_pipeline(model_id: str):
    global _PIPELINE, _PIPELINE_DEVICE
    if _PIPELINE is not None:
        return _PIPELINE, _PIPELINE_DEVICE
    try:
        import torch
        from diffusers import StableDiffusionPipeline
    except ImportError as exc:
        message = (
            "Missing AI dependencies. Install with:\n"
            "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118\n"
            "  pip install diffusers transformers accelerate safetensors"
        )
        raise RuntimeError(message) from exc
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    pipe = pipe.to(device)
    _PIPELINE = pipe
    _PIPELINE_DEVICE = device
    return pipe, device


def generate_image(
    *,
    prompt: str,
    negative_prompt: str,
    size: int,
    seed: int,
    model_id: str,
    steps: int,
    guidance: float,
    style_preset: str,
) -> Optional[Path]:
    cache_dir = OUTPUT_DIR / "ai_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    sizes = [size]
    for fallback in (448, 384):
        if fallback not in sizes and fallback < size:
            sizes.append(fallback)
    allow_testsrc2 = os.getenv("MONEYOS_ALLOW_TESTSRC2") == "1"
    for attempt_size in sizes:
        cache_key = cache_key_for_prompt(
            prompt,
            negative_prompt,
            size=attempt_size,
            seed=seed,
            style_preset=style_preset,
            model_id=model_id,
        )
        output_path = cache_dir / f"{cache_key}.png"
        if output_path.exists():
            return output_path
        pipe, device = _load_pipeline(model_id)
        try:
            import torch

            generator = torch.Generator(device=device).manual_seed(seed)
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                height=attempt_size,
                width=attempt_size,
                generator=generator,
                num_images_per_prompt=1,
            )
            image = result.images[0]
            image.save(output_path)
            print(f"[AI_VISUALS] generated size={attempt_size} device={device} cache={output_path.name}")
            return output_path
        except Exception as exc:
            message = str(exc).lower()
            if "out of memory" in message and "cuda" in message:
                print(f"[AI_VISUALS] CUDA OOM at size {attempt_size}, retrying smaller size")
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception:
                    pass
                continue
            if allow_testsrc2:
                print(f"[AI_VISUALS] generation failed: {exc}")
                return None
            raise RuntimeError("AI image generation failed.") from exc
    if allow_testsrc2:
        return None
    raise RuntimeError("AI image generation failed after size fallbacks.")
