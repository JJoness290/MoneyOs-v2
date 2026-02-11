from __future__ import annotations

from app.core.visuals.anime_3d.shaders.toon_shader import shader_style_defaults


def variation_to_material_params(variation: dict) -> dict:
    defaults = shader_style_defaults()
    params = {
        "shadow_hardness": defaults["shadow_hardness"],
        "rim_boost": defaults["rim_boost"],
        "hair_color": variation.get("hair_color"),
        "eye_color": variation.get("eye_color"),
        "clothing_color": variation.get("clothing_color"),
        "skin_tone_shift": variation.get("skin_tone_shift", 0.0),
    }
    return params
