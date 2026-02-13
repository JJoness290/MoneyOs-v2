from __future__ import annotations

from app.config import resolve_offline_mode, resolve_sd_disabled, resolve_style_preset, resolve_texture_mode


def test_offline_forces_procedural_and_local(monkeypatch) -> None:
    monkeypatch.setenv("MONEYOS_TEXTURE_MODE", "sd_local")
    monkeypatch.setenv("MONEYOS_STYLE_PRESET", "key_art")
    monkeypatch.setenv("MONEYOS_SD_DISABLE", "0")
    monkeypatch.setenv("MONEYOS_NO_NETWORK", "1")
    monkeypatch.setenv("MONEYOS_DISABLE_NET", "1")

    assert resolve_offline_mode() is True
    assert resolve_sd_disabled() is True
    assert resolve_texture_mode() == "procedural"
    assert resolve_style_preset() == "local"


def test_env_overrides_when_online(monkeypatch) -> None:
    monkeypatch.setenv("MONEYOS_NO_NETWORK", "0")
    monkeypatch.setenv("MONEYOS_DISABLE_NET", "0")
    monkeypatch.setenv("MONEYOS_SD_DISABLE", "1")
    monkeypatch.setenv("MONEYOS_TEXTURE_MODE", "none")
    monkeypatch.setenv("MONEYOS_STYLE_PRESET", "default")

    assert resolve_offline_mode() is False
    assert resolve_sd_disabled() is True
    assert resolve_texture_mode() == "none"
    assert resolve_style_preset() == "default"
