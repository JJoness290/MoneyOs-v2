from __future__ import annotations

import os

from app.config import (
    SD_PROFILE,
    SD_MODEL_ID,
    SD_MODEL_SOURCE,
    SD_MODEL_LOCAL_PATH,
    SD_MAX_BATCH_SIZE,
)
from app.core.visuals.ffmpeg_utils import encoder_self_check, select_video_encoder


def _log(message: str) -> None:
    print(message, flush=True)


def main() -> None:
    _log("[SELF_CHECK] starting")
    try:
        import torch  # noqa: WPS433

        cuda_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"
        _log(
            f"[SELF_CHECK] torch={torch.__version__} cuda_available={cuda_available} gpu={gpu_name}"
        )
    except Exception as exc:  # noqa: BLE001
        _log(f"[SELF_CHECK] torch unavailable ({exc})")

    _log(
        "[SELF_CHECK] sd_profile "
        f"profile={SD_PROFILE} model_source={SD_MODEL_SOURCE} "
        f"model_id={SD_MODEL_ID} local_path={SD_MODEL_LOCAL_PATH} "
        f"max_batch_size={SD_MAX_BATCH_SIZE}"
    )
    _log(f"[SELF_CHECK] encoder_self_check={encoder_self_check()}")
    args, encoder = select_video_encoder()
    _log(f"[SELF_CHECK] encoder={encoder} args={' '.join(args)}")
    _log(
        "[SELF_CHECK] env "
        f"MONEYOS_USE_GPU={os.getenv('MONEYOS_USE_GPU', '0')} "
        f"MONEYOS_NVENC_CODEC={os.getenv('MONEYOS_NVENC_CODEC', 'h264')} "
        f"MONEYOS_NVENC_QUALITY={os.getenv('MONEYOS_NVENC_QUALITY', 'balanced')} "
        f"MONEYOS_NVENC_MODE={os.getenv('MONEYOS_NVENC_MODE', 'cq')}"
    )


if __name__ == "__main__":
    main()
