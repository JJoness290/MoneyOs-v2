import inspect
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tools.bootstrap_repo  # noqa: E402

from app.core.visuals.ai_video.backends.cogvideox import CogVideoXBackend


def main() -> int:
    print(f"python={sys.executable}")
    try:
        import torch

        print(f"torch_cuda_available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"torch_cuda_device={torch.cuda.get_device_name(0)}")
    except Exception as exc:  # noqa: BLE001
        print(f"torch_error={exc}")
    backend = CogVideoXBackend()
    backend_path = inspect.getfile(backend.__class__)
    print(f"backend_file={backend_path}")
    print(f"backend_name={backend.name}")
    available = backend.is_available()
    print(f"backend_is_available={available}")
    print(f"model_id={backend.model_id}")
    print(f"env_offload={os.getenv('MONEYOS_COGVIDEOX_OFFLOAD', '1')}")
    print(f"env_fp16={os.getenv('MONEYOS_COGVIDEOX_FP16', '1')}")
    print(f"env_attention_slicing={os.getenv('MONEYOS_COGVIDEOX_ATTENTION_SLICING', '1')}")
    print(f"env_vae_slicing={os.getenv('MONEYOS_COGVIDEOX_VAE_SLICING', '1')}")
    print(f"env_vae_tiling={os.getenv('MONEYOS_COGVIDEOX_VAE_TILING', '1')}")
    print(f"env_num_frames={os.getenv('MONEYOS_COGVIDEOX_NUM_FRAMES', '')}")
    print(f"env_fps={os.getenv('MONEYOS_COGVIDEOX_FPS', '')}")
    print(f"env_height={os.getenv('MONEYOS_COGVIDEOX_HEIGHT', '')}")
    print(f"env_width={os.getenv('MONEYOS_COGVIDEOX_WIDTH', '')}")
    print(f"env_guidance={os.getenv('MONEYOS_COGVIDEOX_GUIDANCE', '')}")
    print(f"env_steps={os.getenv('MONEYOS_COGVIDEOX_STEPS', '')}")
    print(f"env_seed_mode={os.getenv('MONEYOS_COGVIDEOX_SEED_MODE', 'per_clip')}")
    if not available:
        print("Backend unavailable; check diffusers installation.")
        return 1
    try:
        backend.load()
        print("LOAD OK")
        with TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "verify.mp4"
            backend.generate(
                prompt="A sunny park with gentle wind.",
                negative_prompt="blurry, low quality",
                seed=123,
                seconds=1,
                fps=8,
                width=512,
                height=288,
                steps=8,
                guidance=4.0,
                out_path=out_path,
            )
            print(f"GENERATE OK output={out_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"VERIFY FAILED: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
