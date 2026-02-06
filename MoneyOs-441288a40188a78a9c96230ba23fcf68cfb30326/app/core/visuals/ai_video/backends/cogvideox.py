from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import try_to_load_from_cache

from app.core.visuals.ai_video.backends.base import AiVideoBackend, BackendResult, BackendUnavailable


class CogVideoXBackend(AiVideoBackend):
    name = "COGVIDEOX"

    def __init__(self) -> None:
        self.model_id = os.getenv("MONEYOS_COGVIDEOX_MODEL_ID", "zai-org/CogVideoX-5b")
        self._pipe = None
        self._device = "cpu"

    def is_available(self) -> bool:
        cached = try_to_load_from_cache(self.model_id, "model_index.json")
        return cached is not None

    def load(self) -> None:
        if self._pipe is not None:
            return
        if not self.is_available():
            raise BackendUnavailable(
                "CogVideoX model not found in local cache. "
                "Run: python -c \"from huggingface_hub import snapshot_download; "
                f"snapshot_download('{self.model_id}')\""
            )
        if os.getenv("MONEYOS_USE_GPU", "1") != "0":
            import torch

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        import torch
        from diffusers import CogVideoXPipeline

        dtype = torch.bfloat16
        pipe = CogVideoXPipeline.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            local_files_only=True,
        )
        pipe = pipe.to(self._device)
        self._pipe = pipe

    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        seconds: int,
        fps: int,
        width: int,
        height: int,
        steps: int,
        guidance: float,
        out_path: Path,
    ) -> BackendResult:
        self.load()
        import torch
        from diffusers.utils import export_to_video

        if self._pipe is None:
            raise BackendUnavailable("CogVideoX pipeline not loaded")
        generator = torch.Generator(device=self._device).manual_seed(seed)
        frames = self._pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=int(seconds * fps),
            num_inference_steps=steps,
            guidance_scale=guidance,
            height=height,
            width=width,
            generator=generator,
        ).frames[0]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        export_to_video(frames, str(out_path), fps=fps)
        return BackendResult(
            fps=fps,
            resolution=f"{width}x{height}",
            device=self._device,
        )
