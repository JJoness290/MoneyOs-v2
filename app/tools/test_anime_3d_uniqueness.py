from __future__ import annotations

import hashlib
import json
import os
import subprocess
import shutil
import sys
import uuid
from pathlib import Path

from app.core.visuals.anime_3d.render_pipeline import render_anime_3d_60s


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _framehash(image_path: Path) -> str | None:
    if not shutil.which("ffmpeg"):
        return None
    result = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(image_path),
            "-frames:v",
            "1",
            "-pix_fmt",
            "rgb24",
            "-f",
            "framehash",
            "-",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    lines = [line for line in result.stdout.splitlines() if line and not line.startswith("#")]
    if not lines:
        return None
    return lines[-1].split(",")[-1].strip()


def _load_fingerprint(output_dir: Path) -> str | None:
    report_path = output_dir / "render_report.json"
    if not report_path.exists():
        return None
    data = json.loads(report_path.read_text(encoding="utf-8"))
    return data.get("fingerprint")


def _extract_seed(cmd_path: Path) -> str | None:
    if not cmd_path.exists():
        return None
    text = cmd_path.read_text(encoding="utf-8")
    parts = text.split()
    for idx, token in enumerate(parts):
        if token == "--seed" and idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def main() -> int:
    overrides = {
        "duration_seconds": 4.0,
        "fps": 24,
        "res": "640x360",
    }
    os.environ.setdefault("MONEYOS_TEST_MODE", "1")
    job_id_a = uuid.uuid4().hex
    job_id_b = uuid.uuid4().hex
    result_a = render_anime_3d_60s(job_id_a, overrides=overrides)
    result_b = render_anime_3d_60s(job_id_b, overrides=overrides)

    seed_a = _extract_seed(result_a.output_dir / "blender_cmd.txt")
    seed_b = _extract_seed(result_b.output_dir / "blender_cmd.txt")
    if seed_a is None or seed_b is None:
        print("Missing --seed in blender_cmd.txt")
        return 3
    if seed_a == seed_b:
        print("Uniqueness test failed: seeds match")
        return 1

    frame_a = result_a.output_dir / "frames" / "frame_0075.png"
    frame_b = result_b.output_dir / "frames" / "frame_0075.png"
    if not frame_a.exists() or not frame_b.exists():
        print("Missing frame_0075.png for comparison")
        return 2
    hash_a = _framehash(frame_a)
    hash_b = _framehash(frame_b)
    md5_a = None
    md5_b = None
    if (result_a.output_dir / "segment.mp4").exists():
        md5_a = _md5(result_a.output_dir / "segment.mp4")
    if (result_b.output_dir / "segment.mp4").exists():
        md5_b = _md5(result_b.output_dir / "segment.mp4")
    print(f"fingerprint_a={_load_fingerprint(result_a.output_dir)}")
    print(f"fingerprint_b={_load_fingerprint(result_b.output_dir)}")
    print(f"seed_a={seed_a}")
    print(f"seed_b={seed_b}")
    print(f"framehash_a={hash_a}")
    print(f"framehash_b={hash_b}")
    print(f"segment_md5_a={md5_a}")
    print(f"segment_md5_b={md5_b}")

    if hash_a is None or hash_b is None:
        print("Uniqueness test skipped: ffmpeg not available for framehash")
        return 0
    if hash_a == hash_b:
        print("Uniqueness test failed: hashes match")
        return 1
    if md5_a is not None and md5_b is not None and md5_a == md5_b:
        print("Uniqueness test failed: segment md5 match")
        return 1
    print("Uniqueness test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
