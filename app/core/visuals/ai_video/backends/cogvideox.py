from __future__ import annotations

import contextlib
import gc
import json
import os
from pathlib import Path
import re


from app.core.visuals.ai_video.backends.base import AiVideoBackend, BackendResult, BackendUnavailable


class CogVideoXBackend(AiVideoBackend):
    name = "COGVIDEOX"
    _WEIGHT_MARKERS = (
        "model.safetensors",
        "pytorch_model.bin",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    )
    _SHARDED_SAFE_RE = re.compile(r"^model-\d{5}-of-\d{5}\.safetensors$")

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
    def _env_flag_alias(name: str, aliases: list[str], default: str = "1") -> bool:
        value = os.getenv(name)
        if value is None:
            for alias in aliases:
                value = os.getenv(alias)
                if value is not None:
                    break
        if value is None:
            value = default
        return value.strip().lower() not in {"0", "false", "no"}

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


    def _resolve_diffusers_model_ref(self) -> str:
        model_path_env = os.getenv("MONEYOS_COGVIDEOX_MODEL_PATH", "").strip()
        if model_path_env:
            local = Path(model_path_env)
            if local.exists():
                return str(local)
        try:
            from huggingface_hub import snapshot_download
            snapshot = snapshot_download(repo_id=self.model_id, local_files_only=True)
            return str(snapshot)
        except Exception:  # noqa: BLE001
            return self.model_id

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
        raise BackendUnavailable(
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
            print(f"[AI-VIDEO][COGVIDEOX] shard index not generated (missing safetensors): {exc}")
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
        print(f"[AI-VIDEO][COGVIDEOX] generated shard index: {index_path}")

    def load(self) -> None:
        if self._pipe is not None:
            return
        if not self.is_available():
            raise BackendUnavailable("CogVideoX pipeline import failed; see logs.")
        import torch

        use_gpu = os.getenv("MONEYOS_USE_GPU", "1") != "0"
        self._device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        from diffusers import CogVideoXPipeline, DiffusionPipeline

        self._offload_enabled = self._env_flag_alias(
            "MONEYOS_COGVIDEOX_OFFLOAD",
            ["MONEYOS_AI_VIDEO_OFFLOAD"],
            "1",
        )
        self._fp16_enabled = self._env_flag("MONEYOS_COGVIDEOX_FP16", "1")
        self._attention_slicing = self._env_flag("MONEYOS_COGVIDEOX_ATTENTION_SLICING", "1")
        self._vae_slicing = self._env_flag("MONEYOS_COGVIDEOX_VAE_SLICING", "1")
        self._vae_tiling = self._env_flag("MONEYOS_COGVIDEOX_VAE_TILING", "1")

        dtype = torch.float16 if self._fp16_enabled else torch.float32
        self._dtype = "float16" if self._fp16_enabled else "float32"
        model_ref = self._resolve_diffusers_model_ref()
        print(f"[AI-VIDEO][COGVIDEOX] root_files={self._root_file_flags(model_ref)}")
        component_dir = Path(model_ref) / "text_encoder"
        layout = self._weight_layout(component_dir)
        print(f"[AI-VIDEO][COGVIDEOX] text_encoder load layout={layout} path={component_dir}")
        if layout == "sharded-no-index":
            self._ensure_safetensors_index_for_shards(component_dir)
            layout = self._weight_layout(component_dir)
            print(f"[AI-VIDEO][COGVIDEOX] text_encoder layout_after_index={layout}")
        self._validate_component_weights(model_ref, "text_encoder")
        cache_dir = os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
        load_kwargs = {
            "torch_dtype": dtype,
            "use_safetensors": True,
            "local_files_only": True,
            "cache_dir": cache_dir,
        }
        if self._is_diffusers_snapshot(model_ref):
            print(f"[AI-VIDEO][COGVIDEOX] load_path=diffusers_root model_ref={model_ref}")
            pipe = DiffusionPipeline.from_pretrained(model_ref, **load_kwargs)
        else:
            print(f"[AI-VIDEO][COGVIDEOX] load_path=transformers_root model_ref={model_ref}")
            with contextlib.suppress(Exception):
                pipe = CogVideoXPipeline.from_pretrained(model_ref, **load_kwargs)
            if "pipe" not in locals():
                pipe = DiffusionPipeline.from_pretrained(model_ref, **load_kwargs)
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
        width_env = os.getenv("MONEYOS_COGVIDEOX_WIDTH") or os.getenv("MONEYOS_AI_VIDEO_WIDTH")
        height_env = os.getenv("MONEYOS_COGVIDEOX_HEIGHT") or os.getenv("MONEYOS_AI_VIDEO_HEIGHT")
        width = int(width_env) if width_env else 1024
        height = int(height_env) if height_env else 576
        steps = self._env_int("MONEYOS_COGVIDEOX_STEPS", 25)
        guidance = self._env_float("MONEYOS_COGVIDEOX_GUIDANCE", 6.0)
        num_frames_env = os.getenv("MONEYOS_COGVIDEOX_NUM_FRAMES")
        num_frames = int(num_frames_env) if num_frames_env else int(seconds * fps)
        num_frames = max(1, min(48, num_frames))
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
