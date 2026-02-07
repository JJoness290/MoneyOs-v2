from __future__ import annotations

import sys

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
    available = backend.is_available()
    print(f"backend_is_available={available}")
    print(f"model_id={backend.model_id}")
    if not available:
        print("Backend unavailable; check diffusers installation.")
        return 1
    try:
        backend.load()
        print("LOAD OK")
    except Exception as exc:  # noqa: BLE001
        print(f"LOAD FAILED: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
