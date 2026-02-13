from __future__ import annotations

import contextlib
import gc
import os
from pathlib import Path


from app.core.visuals.anime_trueai_video.provider import ClipRequest, ClipResult, TextToVideoProvider


class CogVideoXProvider(TextToVideoProvider):
    name = "cogvideox"

    def __init__(self) -> None:
        self.model_id = os.getenv("MONEYOS_COGVIDEOX_MODEL_ID", "zai-org/CogVideoX-5b")
        self.model_path = os.getenv("MONEYOS_COGVIDEOX_MODEL_PATH", "")
        self.allow_download = os.getenv("MONEYOS_AI_VIDEO_ALLOW_DOWNLOAD", "1") == "1"
        self._pipe = None
        self._device = "cpu"

    def is_available(self) -> bool:
        try:
            import torch  # noqa: F401
            from diffusers import CogVideoXPipeline  # noqa: F401
            return True
        except Exception:
            return False

    def _resolve_model_ref(self) -> str:
        if self.model_path:
            model_dir = Path(self.model_path)
            if model_dir.exists():
                return str(model_dir)
        cache_home = Path(os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
        local_snapshots = list(cache_home.glob(f"hub/models--{self.model_id.replace('/', '--')}/snapshots/*"))
        if local_snapshots:
            return str(sorted(local_snapshots)[-1])
        if not self.allow_download:
            raise RuntimeError("CogVideoX model not found locally and auto-download is disabled")
        from huggingface_hub import snapshot_download
        return snapshot_download(repo_id=self.model_id)


    @staticmethod
    def _is_diffusers_snapshot(model_ref: str) -> bool:
        path = Path(model_ref)
        if not path.exists():
            return False
        return (path / "model_index.json").exists() and not (path / "config.json").exists()

    def _load(self) -> None:
        if self._pipe is not None:
            return
        import torch
        from diffusers import CogVideoXPipeline, DiffusionPipeline

        self._device = "cuda" if torch.cuda.is_available() and os.getenv("MONEYOS_USE_GPU", "1") != "0" else "cpu"
        dtype = torch.bfloat16 if self._device == "cuda" else torch.float32
        model_ref = self._resolve_model_ref()
        if self._is_diffusers_snapshot(model_ref):
            pipe = DiffusionPipeline.from_pretrained(model_ref, torch_dtype=dtype)
        else:
            with contextlib.suppress(Exception):
                pipe = CogVideoXPipeline.from_pretrained(model_ref, torch_dtype=dtype)
            if "pipe" not in locals():
                pipe = DiffusionPipeline.from_pretrained(model_ref, torch_dtype=dtype)
        if self._device == "cuda":
            if hasattr(pipe, "enable_model_cpu_offload"):
                with contextlib.suppress(Exception):
                    pipe.enable_model_cpu_offload()
            else:
                pipe = pipe.to("cuda")
            with contextlib.suppress(Exception):
                pipe.enable_attention_slicing("max")
            with contextlib.suppress(Exception):
                pipe.enable_vae_slicing()
            with contextlib.suppress(Exception):
                pipe.vae.enable_tiling()
        self._pipe = pipe

    def generate(self, request: ClipRequest) -> ClipResult:
        self._load()
        if self._pipe is None:
            raise RuntimeError("CogVideoX pipeline unavailable")
        import torch
        from diffusers.utils import export_to_video

        self._pipe.set_progress_bar_config(disable=True)
        generator = None
        if self._device == "cuda":
            generator = torch.Generator(device="cuda").manual_seed(request.seed)
        else:
            generator = torch.Generator().manual_seed(request.seed)
        num_frames = request.seconds * request.fps
        with torch.autocast("cuda", dtype=torch.bfloat16) if self._device == "cuda" else contextlib.nullcontext():
            result = self._pipe(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                num_inference_steps=request.steps,
                guidance_scale=request.guidance,
                num_frames=num_frames,
                height=request.height,
                width=request.width,
                generator=generator,
            )
        frames = result.frames[0]
        request.out_path.parent.mkdir(parents=True, exist_ok=True)
        export_to_video(frames, str(request.out_path), fps=request.fps)
        if request.out_path.stat().st_size <= 0:
            raise RuntimeError("generated empty clip")
        with contextlib.suppress(Exception):
            if self._device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
        return ClipResult(
            out_path=request.out_path,
            width=request.width,
            height=request.height,
            fps=request.fps,
            duration_s=float(request.seconds),
            backend=self.name,
        )
