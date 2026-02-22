import os
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


from app.core.audio.tts_xtts import configure_xtts_runtime_env, resolve_tts_license_mode


def test_xtts_runtime_env_redirects_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("MONEYOS_TTS_LICENSE", "cpml")
    tracked = {k: os.environ.get(k) for k in ("TTS_HOME", "XDG_CACHE_HOME", "APPDATA", "COQUI_TOS_AGREED")}
    env = configure_xtts_runtime_env(tmp_path)
    assert env["TTS_HOME"].endswith("tts")
    assert env["XDG_CACHE_HOME"] == str(tmp_path)
    assert env["COQUI_TOS_AGREED"] == "1"
    for key, value in tracked.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def test_xtts_runtime_env_requires_license(monkeypatch, tmp_path):
    monkeypatch.delenv("MONEYOS_TTS_LICENSE", raising=False)
    try:
        configure_xtts_runtime_env(tmp_path)
    except MoneyOSValidationError as exc:
        assert "MONEYOS_TTS_LICENSE" in str(exc)
    else:
        raise AssertionError("expected MoneyOSValidationError")


def test_resolve_tts_license_mode(monkeypatch):
    monkeypatch.setenv("MONEYOS_TTS_LICENSE", "cpml")
    assert resolve_tts_license_mode() == "cpml"
