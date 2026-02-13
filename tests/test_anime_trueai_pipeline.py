from __future__ import annotations

from app.core.visuals.anime_trueai_video.pipeline import _anime_prompt, _negative_prompt


def test_anime_prompt_contains_required_style_tokens() -> None:
    prompt = _anime_prompt("hero runs through neon street", "same protagonist", 0)
    assert "Japanese anime film style" in prompt
    assert "cinematic anime lighting" in prompt
    assert "anime movie quality" in prompt


def test_negative_prompt_contains_forbidden_tokens() -> None:
    neg = _negative_prompt()
    for token in ["photorealistic", "3D render", "scribbles", "watermark"]:
        assert token in neg
