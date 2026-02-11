from __future__ import annotations

from typing import TYPE_CHECKING
import importlib

if TYPE_CHECKING:
    from app.core.visuals.anime_3d.render_pipeline import Anime3DResult, render_anime_3d_60s
    from app.core.visuals.anime_3d.validators import ValidationReport

__all__ = ["Anime3DResult", "render_anime_3d_60s", "ValidationReport"]


def __getattr__(name: str):
    if name in {"Anime3DResult", "render_anime_3d_60s"}:
        module = importlib.import_module("app.core.visuals.anime_3d.render_pipeline")
        return getattr(module, name)
    if name == "ValidationReport":
        module = importlib.import_module("app.core.visuals.anime_3d.validators")
        return getattr(module, name)
    raise AttributeError(name)
