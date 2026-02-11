from __future__ import annotations

import argparse

from app.core.visuals.anime_3d.render_pipeline import finalize_anime_3d


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize anime 3D job outputs.")
    parser.add_argument("--job", required=True, help="Job ID to finalize")
    args = parser.parse_args()
    finalize_anime_3d(args.job)


if __name__ == "__main__":
    main()
