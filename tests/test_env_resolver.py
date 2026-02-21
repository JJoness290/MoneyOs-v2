from __future__ import annotations

import pytest

from app.core.env_resolver import get_env_or_default


def test_get_env_or_default_uses_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MONEYOS_SAMPLE", "value")
    value, source = get_env_or_default("MONEYOS_SAMPLE", "fallback")
    assert value == "value"
    assert source == "env:MONEYOS_SAMPLE"


def test_get_env_or_default_uses_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MONEYOS_SAMPLE", raising=False)
    value, source = get_env_or_default("MONEYOS_SAMPLE", "fallback")
    assert value == "fallback"
    assert source == "default"
