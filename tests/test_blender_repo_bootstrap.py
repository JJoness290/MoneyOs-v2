from __future__ import annotations

import sys
import types
from pathlib import Path


def test_bootstrap_repo_path_for_blender(monkeypatch, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "app").mkdir(parents=True)
    (repo / "src").mkdir(parents=True)
    script = repo / "app" / "core" / "visuals" / "anime_3d" / "blender" / "render_segment.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("# dummy", encoding="utf-8")

    fake_bpy = types.ModuleType("bpy")
    fake_bpy.types = types.SimpleNamespace(Scene=object, Object=object)
    fake_mathutils = types.ModuleType("mathutils")
    fake_mathutils.Vector = object
    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)
    monkeypatch.setitem(sys.modules, "mathutils", fake_mathutils)

    monkeypatch.setenv("MONEYOS_REPO_ROOT", str(repo))

    from app.core.visuals.anime_3d.blender.render_segment import _bootstrap_repo_path_for_blender

    if str(repo) in sys.path:
        sys.path.remove(str(repo))

    detected, added = _bootstrap_repo_path_for_blender(script)

    assert detected == repo
    assert added is True
    assert str(repo) in sys.path
