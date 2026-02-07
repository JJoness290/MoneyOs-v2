from __future__ import annotations

import os
from pathlib import Path

from app.core.visuals.ai_video.backends.base import AiVideoBackend, BackendResult, BackendUnavailable


class CogVideoXBackend(AiVideoBackend):
    name = "COGVIDEOX"

    def __init__(self) -> None:
        self.model_id = os.getenv(
            "MONEYOS_COGVIDEOX_MODEL_ID",
            os.getenv("MONEYOS_AI_VIDEO_MODEL_ID", "zai-org/CogVideoX-5b"),
        )
        self._pipe = None
        self._device = "cpu"
        self._dtype = "float32"

    @staticmethod
    def is_available() -> bool:
        try:
            from diffusers import CogVideoXPipeline  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            print(f"[AI-VIDEO] backend=COGVIDEOX import_failed={exc}")
            return False
        return True

    def load(self) -> None:
        if self._pipe is not None:
            return
        if not self.is_available():
            raise BackendUnavailable("CogVideoX pipeline import failed; see logs.")
        if os.getenv("MONEYOS_USE_GPU", "1") != "0":
            import torch

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        from diffusers import CogVideoXPipeline

        import torch

        dtype = torch.float16 if self._device == "cuda" else torch.float32
        self._dtype = "float16" if self._device == "cuda" else "float32"
        pipe = CogVideoXPipeline.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
        )
        pipe = pipe.to(self._device)
        if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception:  # noqa: BLE001
                pass
        if hasattr(pipe, "enable_attention_slicing"):
            try:
                pipe.enable_attention_slicing()
            except Exception:  # noqa: BLE001
                pass
        if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
            try:
                pipe.vae.enable_tiling()
            except Exception:  # noqa: BLE001
                pass
        self._pipe = pipe
        print(
            "[AI-VIDEO] "
            f"backend=COGVIDEOX model_id={self.model_id} device={self._device} dtype={self._dtype}"
        )

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
        from diffusers.utils import export_to_video

        if self._pipe is None:
            raise BackendUnavailable("CogVideoX pipeline not loaded")
        import torch

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
