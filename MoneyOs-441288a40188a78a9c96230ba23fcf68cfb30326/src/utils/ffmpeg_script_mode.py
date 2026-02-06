from __future__ import annotations

import hashlib
import os
from pathlib import Path

from src.utils.cmdlen import estimate_windows_cmd_length
from src.utils.win_paths import safe_join


def maybe_externalize_filter_graph(args: list[str], job_id: str | None = None) -> list[str]:
    if "-filter_complex" not in args:
        return args
    idx = args.index("-filter_complex")
    if idx + 1 >= len(args):
        return args
    filter_graph = args[idx + 1]
    cmd_len = estimate_windows_cmd_length(args)
    should_externalize = cmd_len > 7000 or len(filter_graph) > 2000
    print(f"[FFmpeg] command length={cmd_len}")
    if not should_externalize:
        return args
    token = job_id or os.getenv("MONEYOS_JOB_ID") or hashlib.sha1(filter_graph.encode("utf-8")).hexdigest()[:8]
    script_path = safe_join("p2", "tmp", f"filter_complex_{token}.txt")
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(filter_graph, encoding="utf-8")
    new_args = list(args)
    new_args[idx] = "-filter_complex_script"
    new_args[idx + 1] = str(script_path)
    print(f"[FFmpeg] using -filter_complex_script: {script_path}")
    return new_args
