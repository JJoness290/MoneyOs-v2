from __future__ import annotations

import json
from pathlib import Path

from app.config import OUTPUT_DIR
from app.core.visuals.anime_3d.render_pipeline import render_episode


def main() -> None:
    episode_dir = OUTPUT_DIR / "episodes" / "anime_3d"
    episode_dir.mkdir(parents=True, exist_ok=True)
    episode_path = episode_dir / "episode_60s.json"
    episode_path.write_text(
        json.dumps(
            {
                "duration_seconds": 60,
                "segments": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    result = render_episode(episode_path, "episode_60s.mp4")
    print(f"rendered={result.success} path={result.output_path} message={result.message}")


if __name__ == "__main__":
    main()
