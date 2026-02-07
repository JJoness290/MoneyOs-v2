from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from moviepy.editor import VideoFileClip
from PIL import Image, ImageChops

from app.core.visuals.ai_video.backends.base import AiVideoBackend, BackendUnavailable, BackendResult
from app.core.visuals.ai_video.backends.cogvideox import CogVideoXBackend
from app.core.visuals.ai_video.backends.svd import SvdBackend
from app.core.visuals.ai_video.backends.animatediff import AnimateDiffBackend
from app.core.visuals.ai_video.prompts import PromptPack


class ClipStaticError(RuntimeError):
    pass


@dataclass(frozen=True)
class GenerationResult:
    backend: str
    fps: int
    resolution: str
    device: str
    duration_s: float
    regenerations: int


_BACKEND_CACHE: dict[str, AiVideoBackend] = {}


def _backend_preference() -> list[AiVideoBackend]:
    return [CogVideoXBackend(), SvdBackend(), AnimateDiffBackend()]


def _select_backend() -> AiVideoBackend:
    requested = os.getenv("MONEYOS_AI_VIDEO_BACKEND", "AUTO").strip().upper()
    requested = requested.replace("COGVIDEＯX", "COGVIDEOX")
    options = _backend_preference()
    availability = []
    for backend in options:
        try:
            available = backend.is_available()  # type: ignore[call-arg]
        except Exception as exc:  # noqa: BLE001
            available = False
            print(f"[AI-VIDEO] backend={backend.name} is_available_error={exc}")
        availability.append((backend.name, available))
    print(f"[AI-VIDEO] backend_availability={availability}")
    if requested != "AUTO":
        for backend in options:
            if backend.name == requested:
                if backend.is_available():  # type: ignore[call-arg]
                    return backend
                raise BackendUnavailable(
                    f"Requested backend unavailable: {requested}; is_available returned False; see logs"
                )
        raise BackendUnavailable(f"Unknown backend: {requested}")
    for backend in options:
        if backend.is_available():  # type: ignore[call-arg]
            return backend
    raise BackendUnavailable("No AI video backend available. Install CogVideoX/SVD/AnimateDiff.")


def backend_availability_table() -> dict[str, bool]:
    table: dict[str, bool] = {}
    for backend in _backend_preference():
        try:
            table[backend.name] = bool(backend.is_available())  # type: ignore[call-arg]
        except Exception:  # noqa: BLE001
            table[backend.name] = False
    return table


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
        t_start = 0.0
        t_end = max(0.0, seconds - (1.0 / max(clip.fps, 1)))
        frame_a = clip.get_frame(t_start)
        frame_b = clip.get_frame(t_end)
        duration = float(clip.duration)
    img_a = Image.fromarray(frame_a)
    img_b = Image.fromarray(frame_b)
    score = _mean_abs_diff(img_a, img_b)
    if abs(duration - seconds) > 0.25:
        return False
    return score >= threshold


def generate_clip(
    prompt_pack: PromptPack,
    seed: int,
    seconds: int,
    out_path: Path,
    fps: int,
    width: int,
    height: int,
    steps: int,
    guidance: float,
) -> GenerationResult:
    backend = _select_backend()
    backend_key = backend.name
    if backend_key not in _BACKEND_CACHE:
        _BACKEND_CACHE[backend_key] = backend
    active_backend = _BACKEND_CACHE[backend_key]
    result: BackendResult = active_backend.generate(
        prompt=prompt_pack.prompt,
        negative_prompt=prompt_pack.negative_prompt,
        seed=seed,
        seconds=seconds,
        fps=fps,
        width=width,
        height=height,
        steps=steps,
        guidance=guidance,
        out_path=out_path,
    )
    regenerations = 0
    if not _validate_motion(out_path, seconds):
        regenerations = 1
        regen_seed = seed + 1
        active_backend.generate(
            prompt=prompt_pack.prompt,
            negative_prompt=prompt_pack.negative_prompt,
            seed=regen_seed,
            seconds=seconds,
            fps=fps,
            width=width,
            height=height,
            steps=steps,
            guidance=guidance,
            out_path=out_path,
        )
        if not _validate_motion(out_path, seconds):
            raise ClipStaticError(f"Clip appears static after regeneration: {out_path}")
    device = result.device
    if os.getenv("MONEYOS_USE_GPU", "1") != "0":
        try:
            import torch

            if torch.cuda.is_available() and device != "cuda":
                raise RuntimeError("GPU available but backend did not use CUDA.")
        except ImportError:
            pass
    with VideoFileClip(str(out_path)) as clip:
        duration = float(clip.duration)
    return GenerationResult(
        backend=backend.name,
        fps=result.fps,
        resolution=result.resolution,
        device=device,
        duration_s=duration,
        regenerations=regenerations,
    )
