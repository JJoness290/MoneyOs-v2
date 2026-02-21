from __future__ import annotations

import contextlib
import gc
import json
import os
from pathlib import Path
import re

from app.core.visuals.anime_trueai_video.provider import ClipRequest, ClipResult, TextToVideoProvider
from app.core.stability import resolve_stability_settings


class CogVideoXProvider(TextToVideoProvider):
    name = "cogvideox"
    _shared_pipe = None
    _shared_device = "cpu"
    _WEIGHT_MARKERS = (
        "model.safetensors",
        "pytorch_model.bin",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    )
    _SHARDED_SAFE_RE = re.compile(r"^model-\d{5}-of-\d{5}\.safetensors$")

    def __init__(self) -> None:
        self.model_id = os.getenv("MONEYOS_COGVIDEOX_MODEL_ID", "zai-org/CogVideoX-5b")
        self.model_path = os.getenv("MONEYOS_COGVIDEOX_MODEL_PATH", "")
        self.allow_download = os.getenv("MONEYOS_AI_VIDEO_ALLOW_DOWNLOAD", "1") == "1"
        self._pipe = CogVideoXProvider._shared_pipe
        self._device = CogVideoXProvider._shared_device
        self._generation_calls = 0
        self._compiled = False

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
            # Prefer snapshots with a usable text_encoder weight marker.
            for snap in sorted(local_snapshots, reverse=True):
                if self._has_usable_weight_file(snap / "text_encoder"):
                    return str(snap)
            return str(sorted(local_snapshots)[-1])
        if not self.allow_download:
            raise RuntimeError("CogVideoX model not found locally and auto-download is disabled")
        from huggingface_hub import snapshot_download

        return snapshot_download(repo_id=self.model_id)

    @classmethod
    def _has_usable_weight_file(cls, model_dir: Path) -> bool:
        if not model_dir.exists() or not model_dir.is_dir():
            return False
        if any((model_dir / marker).exists() for marker in cls._WEIGHT_MARKERS):
            return True
        return any(cls._SHARDED_SAFE_RE.match(p.name) for p in model_dir.iterdir() if p.is_file())

    @classmethod
    def _weight_layout(cls, model_dir: Path) -> str:
        if not model_dir.exists() or not model_dir.is_dir():
            return "missing"
        if (model_dir / "model.safetensors").exists() or (model_dir / "pytorch_model.bin").exists():
            return "single-file"
        if (model_dir / "model.safetensors.index.json").exists() or (model_dir / "pytorch_model.bin.index.json").exists():
            return "indexed-sharded"
        if any(cls._SHARDED_SAFE_RE.match(p.name) for p in model_dir.iterdir() if p.is_file()):
            return "sharded-no-index"
        return "unknown"

    @staticmethod
    def _list_dir_files(model_dir: Path) -> list[str]:
        if not model_dir.exists() or not model_dir.is_dir():
            return []
        return sorted([entry.name for entry in model_dir.iterdir() if entry.is_file()])

    def _validate_component_weights(self, model_ref: str, component: str = "text_encoder") -> None:
        component_dir = Path(model_ref) / component
        if not component_dir.exists():
            return
        if self._has_usable_weight_file(component_dir):
            return
        files = self._list_dir_files(component_dir)
        raise RuntimeError(
            "CogVideoX load failed: component directory does not contain a usable weight file "
            f"({', '.join(self._WEIGHT_MARKERS)}). component={component} path={component_dir} files={files}"
        )

    def _ensure_safetensors_index_for_shards(self, model_dir: Path) -> None:
        index_path = model_dir / "model.safetensors.index.json"
        if index_path.exists():
            return
        shard_files = sorted([p for p in model_dir.iterdir() if p.is_file() and self._SHARDED_SAFE_RE.match(p.name)])
        if not shard_files:
            return
        try:
            from safetensors import safe_open  # type: ignore
        except Exception as exc:  # noqa: BLE001
            print(f"[TRUEAI][COGVIDEOX] shard index not generated (missing safetensors): {exc}")
            return
        weight_map: dict[str, str] = {}
        for shard in shard_files:
            with safe_open(str(shard), framework="pt", device="cpu") as sf:
                for key in sf.keys():
                    weight_map[key] = shard.name
        payload = {
            "metadata": {"total_size": sum(p.stat().st_size for p in shard_files)},
            "weight_map": weight_map,
        }
        index_path.write_text(json.dumps(payload), encoding="utf-8")
        print(f"[TRUEAI][COGVIDEOX] generated shard index: {index_path}")

    @staticmethod
    def _is_diffusers_snapshot(model_ref: str) -> bool:
        path = Path(model_ref)
        if not path.exists():
            return False
        return (path / "model_index.json").exists()

    @staticmethod
    def _root_file_flags(model_ref: str) -> dict[str, bool]:
        root = Path(model_ref)
        return {
            "model_index.json": (root / "model_index.json").exists(),
            "configuration.json": (root / "configuration.json").exists(),
            "config.json": (root / "config.json").exists(),
        }

    @staticmethod
    def detect_model_format(model_ref: str) -> str:
        root = Path(model_ref)
        if (root / "model_index.json").exists():
            return "diffusers"
        if (root / "config.json").exists():
            return "transformers"
        return "transformers"

    def _load(self) -> None:
        if self._pipe is not None:
            return
        import torch
        from diffusers import CogVideoXPipeline, DiffusionPipeline

        self._device = "cuda" if torch.cuda.is_available() and os.getenv("MONEYOS_USE_GPU", "1") != "0" else "cpu"
        dtype = torch.float16 if self._device == "cuda" else torch.float32

        with contextlib.suppress(Exception):
            torch.backends.cuda.matmul.allow_tf32 = True
        with contextlib.suppress(Exception):
            torch.backends.cudnn.allow_tf32 = True
        with contextlib.suppress(Exception):
            torch.set_float32_matmul_precision("high")

        model_ref = self._resolve_model_ref()
        print(f"[TRUEAI][COGVIDEOX] root_files={self._root_file_flags(model_ref)}")
        component_dir = Path(model_ref) / "text_encoder"
        layout = self._weight_layout(component_dir)
        print(f"[TRUEAI][COGVIDEOX] text_encoder load layout={layout} path={component_dir}")
        if layout == "sharded-no-index":
            self._ensure_safetensors_index_for_shards(component_dir)
            layout = self._weight_layout(component_dir)
            print(f"[TRUEAI][COGVIDEOX] text_encoder layout_after_index={layout}")
        self._validate_component_weights(model_ref, "text_encoder")
        cache_dir = os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
        load_kwargs = {
            "torch_dtype": dtype,
            "use_safetensors": True,
            "local_files_only": True,
            "cache_dir": cache_dir,
        }
        model_format = self.detect_model_format(model_ref)
        if model_format == "diffusers":
            print(f"[TRUEAI][COGVIDEOX] load_path=diffusers_root model_ref={model_ref}")
            pipe = DiffusionPipeline.from_pretrained(model_ref, **load_kwargs)
        else:
            if Path(model_ref).exists() and (Path(model_ref) / "model_index.json").exists():
                root_files = self._list_dir_files(Path(model_ref))
                raise RuntimeError(
                    "BUG: attempted transformers load on diffusers root "
                    f"path={model_ref} files={root_files}"
                )
            print(f"[TRUEAI][COGVIDEOX] load_path=transformers_root model_ref={model_ref}")
            with contextlib.suppress(Exception):
                pipe = CogVideoXPipeline.from_pretrained(model_ref, **load_kwargs)
            if "pipe" not in locals():
                pipe = DiffusionPipeline.from_pretrained(model_ref, **load_kwargs)

        with contextlib.suppress(Exception):
            from diffusers import DPMSolverMultistepScheduler  # noqa: WPS433

            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

        if self._device == "cuda":
            stability = resolve_stability_settings()
            if stability.stability_mode:
                fraction = max(0.3, min(0.95, stability.vram_fraction))
                with contextlib.suppress(Exception):
                    torch.cuda.memory.set_per_process_memory_fraction(fraction, device=0)
                print(f"[TRUEAI][COGVIDEOX] vram_guardrail fraction={fraction:.2f}")
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
            with contextlib.suppress(Exception):
                pipe.enable_xformers_memory_efficient_attention()

        self._pipe = pipe
        CogVideoXProvider._shared_pipe = pipe
        CogVideoXProvider._shared_device = self._device

    @staticmethod
    def _is_oom_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return "out of memory" in text or "cuda" in text and "memory" in text

    def _stability_degrade(self, width: int, height: int, steps: int, frames: int) -> tuple[int, int, int, int]:
        degraded_w = max(384, ((int(width * 0.85)) // 8) * 8)
        degraded_h = max(216, ((int(height * 0.85)) // 8) * 8)
        degraded_steps = max(8, steps - 4)
        degraded_frames = max(8, frames - 8)
        return degraded_w, degraded_h, degraded_steps, degraded_frames

    def _maybe_compile(self) -> None:
        if self._compiled or self._pipe is None or self._device != "cuda":
            return
        if os.getenv("MONEYOS_TRUEAI_COMPILE", "0") != "1":
            return
        compile_after_first = os.getenv("MONEYOS_TRUEAI_COMPILE_AFTER_FIRST", "1") == "1"
        if compile_after_first and self._generation_calls < 1:
            return
        import torch

        target = getattr(self._pipe, "transformer", None)
        if target is None:
            return
        try:
            self._pipe.transformer = torch.compile(target, mode="reduce-overhead", fullgraph=False)
            self._compiled = True
            print("[TRUEAI] compile=enabled target=transformer")
        except Exception as exc:  # noqa: BLE001
            print(f"[TRUEAI] compile=failed reason={exc}")

    def generate(self, request: ClipRequest) -> ClipResult:
        self._load()
        if self._pipe is None:
            raise RuntimeError("CogVideoX pipeline unavailable")
        import torch
        from diffusers.utils import export_to_video

        self._maybe_compile()
        self._pipe.set_progress_bar_config(disable=True)
        generator = torch.Generator(device="cuda").manual_seed(request.seed) if self._device == "cuda" else torch.Generator().manual_seed(request.seed)
        with contextlib.suppress(Exception):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        num_frames = int(max(1, min(48, round(request.seconds * request.fps))))
        width = request.width
        height = request.height
        steps = request.steps
        stability = resolve_stability_settings()
        max_attempts = 4 if stability.stability_mode else 1
        last_exc: Exception | None = None
        for attempt in range(max_attempts):
            try:
                if self._device == "cuda" and stability.stability_mode:
                    total = float(torch.cuda.get_device_properties(0).total_memory)
                    reserved = float(torch.cuda.memory_reserved(0))
                    util = reserved / max(total, 1.0)
                    if util >= min(0.99, stability.vram_fraction + 0.05) and attempt < (max_attempts - 1):
                        width, height, steps, num_frames = self._stability_degrade(width, height, steps, num_frames)
                        print(
                            "[TRUEAI][COGVIDEOX] guardrail_trigger="
                            f"reserved_ratio({util:.3f})>=budget({stability.vram_fraction:.3f}) "
                            f"degrade_to={width}x{height} steps={steps} frames={num_frames}"
                        )
                        continue
                with torch.autocast("cuda", dtype=torch.float16) if self._device == "cuda" else contextlib.nullcontext():
                    result = self._pipe(
                        prompt=request.prompt,
                        negative_prompt=request.negative_prompt,
                        num_inference_steps=steps,
                        guidance_scale=request.guidance,
                        num_frames=num_frames,
                        height=height,
                        width=width,
                        generator=generator,
                    )
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if not stability.stability_mode or not self._is_oom_error(exc) or attempt >= (max_attempts - 1):
                    raise
                width, height, steps, num_frames = self._stability_degrade(width, height, steps, num_frames)
                with contextlib.suppress(Exception):
                    if hasattr(self._pipe, "enable_model_cpu_offload"):
                        self._pipe.enable_model_cpu_offload()
                    if hasattr(self._pipe, "enable_attention_slicing"):
                        self._pipe.enable_attention_slicing("max")
                    if hasattr(self._pipe, "enable_vae_slicing"):
                        self._pipe.enable_vae_slicing()
                print(
                    "[TRUEAI][COGVIDEOX] oom_degrade "
                    f"attempt={attempt + 1}/{max_attempts} to={width}x{height} steps={steps} frames={num_frames} reason={exc}"
                )
        else:
            raise RuntimeError(
                "VRAM guardrail blocked generation after automatic degradation. "
                f"budget_fraction={stability.vram_fraction} last_error={last_exc}"
            )
        frames = result.frames[0]
        request.out_path.parent.mkdir(parents=True, exist_ok=True)
        export_to_video(frames, str(request.out_path), fps=request.fps)
        if request.out_path.stat().st_size <= 0:
            raise RuntimeError("generated empty clip")
        self._generation_calls += 1
        with contextlib.suppress(Exception):
            if self._device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
        return ClipResult(
            out_path=request.out_path,
            width=request.width,
            height=request.height,
            fps=request.fps,
            duration_s=float(num_frames / request.fps),
            backend=self.name,
        )
