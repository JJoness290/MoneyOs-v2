from __future__ import annotations

import os
from pathlib import Path
import shutil


def _bytes_to_gb(value: int) -> float:
    return value / (1024**3)


def ensure_storage_budget(
    paths: list[Path],
    required_bytes: int,
    stage: str,
) -> None:
    required_bytes = max(0, int(required_bytes))
    for path in paths:
        root = path.resolve()
        if root.is_file():
            root = root.parent
        usage = shutil.disk_usage(root)
        if usage.free < required_bytes:
            print(
                "[STORAGE] insufficient disk space; stopping downloads and proceeding with best assets acquired",
                flush=True,
            )
            raise RuntimeError(
                f"Insufficient disk space for {stage}. "
                f"need={_bytes_to_gb(required_bytes):.2f}GB free={_bytes_to_gb(usage.free):.2f}GB "
                f"path={root}"
            )


def compute_required_bytes(
    *,
    estimated_download_size: int = 0,
    extraction_overhead: int = 0,
    render_temp_budget: int = 0,
    final_output_estimate: int = 0,
) -> int:
    return int(estimated_download_size + extraction_overhead + render_temp_budget + final_output_estimate)


def default_render_budget_bytes() -> int:
    return int(os.getenv("MONEYOS_RENDER_TEMP_BUDGET_BYTES", str(5 * 1024**3)))


def default_output_estimate_bytes() -> int:
    return int(os.getenv("MONEYOS_OUTPUT_ESTIMATE_BYTES", str(2 * 1024**3)))
