from app.core.story.anime_writer import generate_anime_episode_outline_and_script


FORBIDDEN = ["Narration:", "Scene 1", "Beat 2", "pushes the conflict"]


def test_writer_generates_natural_dialogue_without_placeholders():
    script = generate_anime_episode_outline_and_script("Rogue AI", 1)
    all_dialogue = "\n".join(
        beat["dialogue"]
        for scene in script["scenes"]
        for beat in scene["beats"]
    )
    for token in FORBIDDEN:
        assert token not in all_dialogue
    assert "Ren Aoki:" in all_dialogue
    assert "Mika Sora:" in all_dialogue
