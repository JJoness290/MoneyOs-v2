from __future__ import annotations

import os
from pathlib import Path

from app.core.visuals.anime_3d.render_pipeline import render_anime_3d_60s


def main() -> None:
    os.environ.setdefault("MONEYOS_TEST_MODE", "1")
    job_id = "duration-override-test"
    result = render_anime_3d_60s(job_id, overrides={"duration_s": 8, "fps": 24})
    cmd_path = result.output_dir / "blender_cmd.txt"
    cmd_text = cmd_path.read_text(encoding="utf-8")
    if "--duration 8.000000" not in cmd_text:
        raise AssertionError("blender_cmd.txt missing overridden duration")
    if "--fps 24" not in cmd_text:
        raise AssertionError("blender_cmd.txt missing overridden fps")
    print(f"OK: {cmd_path}")


if __name__ == "__main__":
    main()
