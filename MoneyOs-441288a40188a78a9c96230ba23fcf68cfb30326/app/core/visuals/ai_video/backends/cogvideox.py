from __future__ import annotations

import contextlib
import gc
import os
from pathlib import Path

from app.core.visuals.ai_video.backends.base import AiVideoBackend, BackendResult, BackendUnavailable


class CogVideoXBackend(AiVideoBackend):
    name = "COGVIDEOX"

    def __init__(self) -> None:
        self.model_id = os.getenv(
            "MONEYOS_AI_VIDEO_MODEL_ID",
            os.getenv("MONEYOS_COGVIDEOX_MODEL_ID", "zai-org/CogVideoX-5b"),
        )
        self._pipe = None
        self._device = "cpu"
        self._dtype = "float32"
        self._offload_enabled = True
        self._fp16_enabled = True
        self._attention_slicing = True
        self._vae_slicing = True
        self._vae_tiling = True

    @staticmethod
    def _env_flag(name: str, default: str = "1") -> bool:
        return os.getenv(name, default).strip().lower() not in {"0", "false", "no"}

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        value = os.getenv(name)
        if value is None or value == "":
            return default
        return int(value)

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        value = os.getenv(name)
        if value is None or value == "":
            return default
        return float(value)

    def is_available(self) -> bool:
        try:
            from diffusers import CogVideoXPipeline  # noqa: F401
            import torch  # noqa: F401
            return True
        except Exception as e:  # noqa: BLE001
            print(f"[CogVideoXBackend] is_available failed: {e}")
            return False

    def load(self) -> None:
        if self._pipe is not None:
            return
        if not self.is_available():
            raise BackendUnavailable("CogVideoX pipeline import failed; see logs.")
        import torch

        use_gpu = os.getenv("MONEYOS_USE_GPU", "1") != "0"
        self._device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        from diffusers import CogVideoXPipeline

        self._offload_enabled = self._env_flag("MONEYOS_COGVIDEOX_OFFLOAD", "1")
        self._fp16_enabled = self._env_flag("MONEYOS_COGVIDEOX_FP16", "1")
        self._attention_slicing = self._env_flag("MONEYOS_COGVIDEOX_ATTENTION_SLICING", "1")
        self._vae_slicing = self._env_flag("MONEYOS_COGVIDEOX_VAE_SLICING", "1")
        self._vae_tiling = self._env_flag("MONEYOS_COGVIDEOX_VAE_TILING", "1")

        dtype = torch.float16 if self._fp16_enabled else torch.float32
        self._dtype = "float16" if self._fp16_enabled else "float32"
        pipe = CogVideoXPipeline.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
        )
        if self._device == "cuda" and not self._offload_enabled:
            pipe = pipe.to(self._device)
        if self._device == "cuda" and self._offload_enabled and hasattr(pipe, "enable_model_cpu_offload"):
            try:
                pipe.enable_model_cpu_offload()
            except Exception:  # noqa: BLE001
                pass
        if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception:  # noqa: BLE001
                pass
        if self._attention_slicing and hasattr(pipe, "enable_attention_slicing"):
            try:
                pipe.enable_attention_slicing("max")
            except Exception:  # noqa: BLE001
                pass
        if self._vae_slicing and hasattr(pipe, "enable_vae_slicing"):
            try:
                pipe.enable_vae_slicing()
            except Exception:  # noqa: BLE001
                pass
        if (
            self._vae_tiling
            and hasattr(pipe, "vae")
            and hasattr(pipe.vae, "enable_tiling")
        ):
            try:
                pipe.vae.enable_tiling()
            except Exception:  # noqa: BLE001
                pass
        self._pipe = pipe
        print(
            "[AI-VIDEO] "
            "backend=COGVIDEOX "
            f"model_id={self.model_id} "
            f"device={self._device} "
            f"dtype={self._dtype} "
            f"offload={self._offload_enabled} "
            f"attention_slicing={self._attention_slicing} "
            f"vae_slicing={self._vae_slicing} "
            f"vae_tiling={self._vae_tiling}"
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

        fps = self._env_int("MONEYOS_COGVIDEOX_FPS", 8)
        width = self._env_int("MONEYOS_COGVIDEOX_WIDTH", 1024)
        height = self._env_int("MONEYOS_COGVIDEOX_HEIGHT", 576)
        steps = self._env_int("MONEYOS_COGVIDEOX_STEPS", 25)
        guidance = self._env_float("MONEYOS_COGVIDEOX_GUIDANCE", 6.0)
        num_frames_env = os.getenv("MONEYOS_COGVIDEOX_NUM_FRAMES")
        num_frames = int(num_frames_env) if num_frames_env else int(seconds * fps)
        seed_mode = os.getenv("MONEYOS_COGVIDEOX_SEED_MODE", "per_clip").strip().lower()
        if seed_mode == "fixed":
            seed = int(os.getenv("MONEYOS_COGVIDEOX_SEED", str(seed)))

        generator_device = "cuda" if self._device == "cuda" and torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(seed)
        autocast = (
            torch.autocast("cuda", dtype=torch.float16)
            if self._device == "cuda"
            else contextlib.nullcontext()
        )
        try:
            with torch.inference_mode(), autocast:
                outputs = self._pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_frames=num_frames,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    height=height,
                    width=width,
                    generator=generator,
                )
                frames = outputs.frames[0]
        except torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError(
                "CUDA out of memory while running CogVideoX. "
                "Enable MONEYOS_COGVIDEOX_OFFLOAD=1, reduce MONEYOS_COGVIDEOX_NUM_FRAMES, "
                "reduce resolution, or set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True."
            ) from exc
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise RuntimeError(
                    "CUDA out of memory while running CogVideoX. "
                    "Enable MONEYOS_COGVIDEOX_OFFLOAD=1, reduce MONEYOS_COGVIDEOX_NUM_FRAMES, "
                    "reduce resolution, or set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True."
                ) from exc
            raise
        out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[AI_VIDEO][COGVIDEOX] exporting to: {out_path}", flush=True)
        try:
            export_to_video(frames, str(out_path), fps=fps)
            if (not out_path.exists()) or out_path.stat().st_size == 0:
                raise RuntimeError(
                    "[AI_VIDEO][COGVIDEOX] export finished but file missing/empty: "
                    f"{out_path}"
                )
            print(
                f"[AI_VIDEO][COGVIDEOX] wrote: {out_path} bytes={out_path.stat().st_size}",
                flush=True,
            )
        finally:
            if "frames" in locals():
                del frames
            if "outputs" in locals():
                del outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        return BackendResult(
            fps=fps,
            resolution=f"{width}x{height}",
            device=self._device,
        )
