from __future__ import annotations

import os

import pytest

from app.core.storage_policy import StoragePolicyError, resolve_storage_policy


def test_storage_policy_defaults_to_c_moneyos(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "MONEYOS_ROOT",
        "MONEYOS_ASSETS_ROOT",
        "MONEYOS_OUTPUT_ROOT",
        "MONEYOS_CACHE_ROOT",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "HF_DATASETS_CACHE",
        "TORCH_HOME",
        "TRANSFORMERS_CACHE",
        "MONEYOS_TEMP_ROOT",
    ):
        monkeypatch.delenv(key, raising=False)
    policy = resolve_storage_policy()
    assert policy.root.value == r"C:\MoneyOS"
    assert policy.output_root.value == r"C:\MoneyOS/work"
    assert policy.hf_hub_cache.value.lower().startswith(r"c:\moneyos")


def test_storage_policy_blocks_d_drive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HOME", r"D:\MoneyOS\cache\huggingface")
    with pytest.raises(StoragePolicyError):
        resolve_storage_policy()
