from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Callable

from moviepy.editor import VideoFileClip
from PIL import Image, ImageChops

from src.moneyos.ai_video.prompts import negative_prompt


class BackendUnavailable(RuntimeError):
    pass


class ClipStaticError(RuntimeError):
    pass


def _find_spec(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _require_module(module_name: str, install_hint: str) -> None:
    if not _find_spec(module_name):
        raise BackendUnavailable(f"Missing dependency: {module_name}. Install {install_hint}.")


def _load_svd_pipeline(model_path: Path, device: str):
    _require_module("torch", "torch")
    _require_module("diffusers", "diffusers")
    import torch
    from diffusers import StableVideoDiffusionPipeline

    pipe = StableVideoDiffusionPipeline.from_pretrained(
        str(model_path),
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    pipe = pipe.to(device)
    return pipe


def _load_animatediff_pipeline(model_path: Path, device: str):
    _require_module("torch", "torch")
    _require_module("diffusers", "diffusers")
    import torch
    from diffusers import AnimateDiffPipeline

    pipe = AnimateDiffPipeline.from_pretrained(
        str(model_path),
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    pipe = pipe.to(device)
    return pipe


def _device() -> str:
    if os.getenv("MONEYOS_USE_GPU", "1") == "0":
        return "cpu"
    if not _find_spec("torch"):
        return "cpu"
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _select_backend() -> tuple[str, Path]:
    backend = os.getenv("MONEYOS_AI_VIDEO_BACKEND", "AUTO").strip().upper()
    svd_path = Path(os.getenv("MONEYOS_SVD_MODEL_PATH", "models/svd"))
    anim_path = Path(os.getenv("MONEYOS_ANIMATEDIFF_MODEL_PATH", "models/animatediff"))
    if backend == "SVD":
        return "SVD", svd_path
    if backend == "ANIMATEDIFF":
        return "ANIMATEDIFF", anim_path
    if svd_path.exists():
        return "SVD", svd_path
    if anim_path.exists():
        return "ANIMATEDIFF", anim_path
    raise BackendUnavailable("No AI video backend available (SVD or AnimateDiff model missing).")


_PIPELINE_CACHE: dict[str, object] = {}


def _get_pipeline(backend: str, model_path: Path, device: str):
    cache_key = f"{backend}:{model_path.resolve()}:{device}"
    if cache_key in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[cache_key]
    if backend == "SVD":
        pipe = _load_svd_pipeline(model_path, device)
    elif backend == "ANIMATEDIFF":
        pipe = _load_animatediff_pipeline(model_path, device)
    else:
        raise BackendUnavailable(f"Unsupported backend: {backend}")
    _PIPELINE_CACHE[cache_key] = pipe
    return pipe


def _mean_abs_diff(img_a: Image.Image, img_b: Image.Image) -> float:
    diff = ImageChops.difference(img_a, img_b)
    histogram = diff.histogram()
    total_pixels = img_a.size[0] * img_a.size[1]
    value = 0.0
    for i, count in enumerate(histogram):
        value += (i % 256) * count
    return value / max(total_pixels, 1)


def _validate_motion(video_path: Path, seconds: float, threshold: float = 6.0) -> bool:
    with VideoFileClip(str(video_path)) as clip:
        t_start = min(0.2, max(0.0, seconds - 0.2))
        t_end = max(0.2, seconds - 0.2)
        frame_a = clip.get_frame(t_start)
        frame_b = clip.get_frame(t_end)
    img_a = Image.fromarray(frame_a)
    img_b = Image.fromarray(frame_b)
    score = _mean_abs_diff(img_a, img_b)
    return score >= threshold


def generate_clip(prompt: str, seed: int, seconds: int, out_path: Path) -> dict[str, object]:
    backend, model_path = _select_backend()
    device = _device()
    fps = int(os.getenv("MONEYOS_AI_FPS", "30"))
    width = int(os.getenv("MONEYOS_AI_WIDTH", "1024"))
    height = int(os.getenv("MONEYOS_AI_HEIGHT", "576"))
    pipe = _get_pipeline(backend, model_path, device)
    _require_module("torch", "torch")
    import torch
    from diffusers.utils import export_to_video

    generator = torch.Generator(device=device).manual_seed(seed)
    negative = negative_prompt()
    num_frames = max(1, int(seconds * fps))
    if backend == "SVD":
        result = pipe(
            prompt=prompt,
            negative_prompt=negative,
            num_frames=num_frames,
            height=height,
            width=width,
            generator=generator,
        )
    else:
        result = pipe(
            prompt=prompt,
            negative_prompt=negative,
            num_frames=num_frames,
            height=height,
            width=width,
            generator=generator,
        )
    frames = result.frames[0] if hasattr(result, "frames") else result
    out_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(out_path), fps=fps)
    if not _validate_motion(out_path, seconds):
        regen_seed = seed + 1
        generator = torch.Generator(device=device).manual_seed(regen_seed)
        result = pipe(
            prompt=prompt,
            negative_prompt=negative,
            num_frames=num_frames,
            height=height,
            width=width,
            generator=generator,
        )
        frames = result.frames[0] if hasattr(result, "frames") else result
        export_to_video(frames, str(out_path), fps=fps)
        if not _validate_motion(out_path, seconds):
            raise ClipStaticError(f"Clip appears static after regeneration: {out_path}")
    return {
        "backend": backend,
        "device": device,
        "fps": fps,
        "resolution": f"{width}x{height}",
    }
