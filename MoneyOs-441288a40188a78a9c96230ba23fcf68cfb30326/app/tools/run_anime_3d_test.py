from __future__ import annotations

import uuid

from app.core.visuals.anime_3d.render_pipeline import render_anime_3d_60s


def main() -> None:
    job_id = uuid.uuid4().hex
    result = render_anime_3d_60s(job_id)
    print(f"rendered=True path={result.final_video} output_dir={result.output_dir}")


if __name__ == "__main__":
    main()
