from app.core.story.sanitizer import sanitize_spoken_script


def test_sanitizer_removes_meta_labels_and_tags():
    text = """
Scene 1
NARRATION: The city burns.
HERO: We move now.
[explosion]
(CUT TO BLACK)
"""
    out = sanitize_spoken_script(text)
    assert "Scene" not in out
    assert "NARRATION" not in out
    assert "HERO:" not in out
    assert "We move now." in out
