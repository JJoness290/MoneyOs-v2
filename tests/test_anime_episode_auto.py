from app.core.story.anime_writer import generate_anime_episode_outline_and_script
from app.core.director.scene_interpreter import build_prompts_and_render_plan


def test_script_schema_has_required_keys(tmp_path):
    payload = generate_anime_episode_outline_and_script("Rogue AI", 10)
    assert payload["title"]
    assert payload["characters"]
    assert payload["scenes"]
    first = payload["scenes"][0]
    assert first["beats"]
    assert "duration_sec_target" in first["beats"][0]


def test_render_plan_duration_matches_audio(tmp_path):
    script = generate_anime_episode_outline_and_script("Neo-Tokyo", 1)
    timestamps = {"total_duration_sec": 60.0, "segments": []}
    _, plan = build_prompts_and_render_plan(script, timestamps, tmp_path)
    total = sum(float(s["duration_sec"]) for s in plan["shots"])
    assert abs(total - 60.0) <= 0.25


from app.core.audio.tts_xtts import MoneyOSValidationError, _normalize_xtts_model_name


def test_xtts_model_alias_accepts_two_fields():
    assert _normalize_xtts_model_name("coqui/XTTS-v2") == "tts_models/multilingual/multi-dataset/xtts_v2"


def test_xtts_model_invalid_two_fields_raises_actionable_error():
    try:
        _normalize_xtts_model_name("bad/model")
    except MoneyOSValidationError as exc:
        assert "MONEYOS_TTS_MODEL" in str(exc)
    else:
        raise AssertionError("expected MoneyOSValidationError")
