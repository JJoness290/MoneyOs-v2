from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.config import OUTPUT_DIR
from app.core.visuals.anime_3d.blender_runner import BlenderCommand, run_blender
from app.core.visuals.anime_3d.validators import validate_render


@dataclass(frozen=True)
class RenderResult:
    output_path: Path
    success: bool
    message: str


def render_episode(episode_json: Path, output_name: str) -> RenderResult:
    output_dir = OUTPUT_DIR / "episodes" / "anime_3d"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_name
    script_path = Path(__file__).parent / "blender" / "render_segment.py"
    try:
        run_blender(
            BlenderCommand(
                script_path=script_path,
                args=[
                    "--episode",
                    str(episode_json),
                    "--output",
                    str(output_path),
                ],
            )
        )
    except Exception as exc:  # noqa: BLE001
        return RenderResult(output_path=output_path, success=False, message=str(exc))

    report = validate_render(output_path)
    if not report.valid:
        return RenderResult(output_path=output_path, success=False, message=report.message)
    return RenderResult(output_path=output_path, success=True, message="rendered")
