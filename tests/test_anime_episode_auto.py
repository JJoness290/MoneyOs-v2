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
