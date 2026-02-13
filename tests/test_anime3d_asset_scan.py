from __future__ import annotations

from pathlib import Path

from app.core.visuals.anime_3d.asset_scan import discover_assets, select_animation_assets, select_character_assets, select_env_blend


def test_asset_scan_and_selection_prefers_local_files(tmp_path: Path) -> None:
    envs = tmp_path / "envs"
    chars = tmp_path / "characters"
    starter = chars / "starter_pack"
    anims = tmp_path / "anims"
    vfx = tmp_path / "vfx"
    for d in (envs, chars, starter, anims, vfx):
        d.mkdir(parents=True, exist_ok=True)

    (envs / "room.blend").write_text("x", encoding="utf-8")
    (envs / "CITY.BLEND").write_text("x", encoding="utf-8")
    (chars / "hero.blend").write_text("x", encoding="utf-8")
    (chars / "enemy.blend").write_text("x", encoding="utf-8")
    (starter / "npc.BLEND").write_text("x", encoding="utf-8")
    (chars / "anime_hero.obj").write_text("x", encoding="utf-8")
    (chars / "anime_hero.mtl").write_text("x", encoding="utf-8")
    (anims / "idle.fbx").write_text("x", encoding="utf-8")
    (anims / "RUN.FBX").write_text("x", encoding="utf-8")
    (anims / "punch.fbx").write_text("x", encoding="utf-8")
    (vfx / "explosion.png").write_text("x", encoding="utf-8")
    (vfx / "energy_arc.PNG").write_text("x", encoding="utf-8")

    inventory, _asset_dirs = discover_assets(tmp_path)
    assert len(inventory["envs"]) >= 2
    assert len(inventory["characters"]) >= 2
    assert len(inventory["anims"]) >= 3
    assert len(inventory["vfx"]) >= 2

    env = select_env_blend(inventory["envs"], "room")
    hero, enemy = select_character_assets(inventory["characters"])
    anims_selected = select_animation_assets(inventory["anims"])

    assert env is not None and env.name.lower() == "room.blend"
    assert hero is not None and hero.name.lower() == "hero.blend"
    assert enemy is not None
    assert anims_selected["idle"] is not None
    assert anims_selected["run"] is not None
    assert anims_selected["punch"] is not None
