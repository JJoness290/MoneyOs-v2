from pathlib import Path

from app.core.audio.voice_provision import ensure_voice_ref_wav


def test_ensure_voice_ref_wav_uses_existing_request_path(tmp_path):
    p = tmp_path / "ref.wav"
    p.write_bytes(b"RIFF")
    assert ensure_voice_ref_wav(str(p)) == p
