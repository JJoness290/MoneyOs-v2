from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download


class BootstrapDownloadError(RuntimeError):
    pass


def ensure_hf_model(repo_id: str, revision: str | None = None, allow_patterns: list[str] | None = None, local_dir: Path | None = None) -> Path:
    target = local_dir or Path(os.getenv("HF_HOME", r"C:\MoneyOS\cache\huggingface")) / repo_id.replace("/", "__")
    target.mkdir(parents=True, exist_ok=True)
    try:
        path = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=allow_patterns,
            local_dir=str(target),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
    except Exception as exc:  # noqa: BLE001
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        if not token:
            raise BootstrapDownloadError(
                f"Failed to download model {repo_id}. If this model is gated, set HF_TOKEN/HUGGINGFACE_TOKEN. Error: {exc}"
            ) from exc
        raise BootstrapDownloadError(f"Failed to download model {repo_id}: {exc}") from exc
    return Path(path)


def ensure_xtts_model() -> Path:
    return ensure_hf_model("coqui/XTTS-v2")


def ensure_trueai_video_model() -> Path:
    repo = os.getenv("MONEYOS_COGVIDEOX_MODEL_ID", "THUDM/CogVideoX-5b")
    return ensure_hf_model(repo)
