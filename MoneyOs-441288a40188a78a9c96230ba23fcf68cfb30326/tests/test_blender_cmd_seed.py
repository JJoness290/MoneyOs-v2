from __future__ import annotations

import json
import os
from pathlib import Path

from app.core.visuals.anime_3d.render_pipeline import render_anime_3d_60s


def _extract_flag(cmd_path: Path, flag: str) -> str | None:
    if not cmd_path.exists():
        return None
    parts = cmd_path.read_text(encoding="utf-8").split()
    for idx, token in enumerate(parts):
        if token == flag and idx + 1 < len(parts):
            return parts[idx + 1].strip('"')
    return None


def _load_report(output_dir: Path) -> dict:
    report_path = output_dir / "render_report.json"
    return json.loads(report_path.read_text(encoding="utf-8"))


def test_blender_cmd_includes_seed_and_fingerprint() -> None:
    os.environ["MONEYOS_TEST_MODE"] = "1"
    overrides = {"duration_seconds": 4.0, "fps": 24, "res": "640x360"}
    result_a = render_anime_3d_60s("episode_seed_a", overrides=overrides)
    result_b = render_anime_3d_60s("episode_seed_b", overrides=overrides)

    cmd_a = result_a.output_dir / "blender_cmd.txt"
    cmd_b = result_b.output_dir / "blender_cmd.txt"
    seed_a = _extract_flag(cmd_a, "--seed")
    seed_b = _extract_flag(cmd_b, "--seed")
    fp_a = _extract_flag(cmd_a, "--fingerprint")
    fp_b = _extract_flag(cmd_b, "--fingerprint")

    assert seed_a is not None
    assert seed_b is not None
    assert fp_a is not None
    assert fp_b is not None
    assert seed_a != seed_b
    assert fp_a != fp_b

    report_a = _load_report(result_a.output_dir)
    report_b = _load_report(result_b.output_dir)
    assert str(report_a.get("seed")) == seed_a
    assert str(report_b.get("seed")) == seed_b
    assert report_a.get("fingerprint") == fp_a
    assert report_b.get("fingerprint") == fp_b
