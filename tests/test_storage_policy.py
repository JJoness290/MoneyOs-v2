from __future__ import annotations

import pytest

from app.core.storage_policy import StoragePolicyError, resolve_storage_policy, validate_storage_path


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
    assert policy.output_root.value == r"C:\MoneyOS\work"
    assert policy.hf_hub_cache.value.lower().startswith(r"c:\moneyos")


def test_storage_policy_blocks_d_drive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HOME", r"D:\MoneyOS\cache\huggingface")
    with pytest.raises(StoragePolicyError):
        resolve_storage_policy()


def test_hf_cache_priority_prefers_hf_hub_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HUB_CACHE", r"C:\MoneyOS\cache\huggingface\hub_custom")
    monkeypatch.setenv("HF_HOME", r"C:\MoneyOS\cache\huggingface_home_custom")
    policy = resolve_storage_policy()
    assert policy.hf_hub_cache.value == r"C:\MoneyOS\cache\huggingface\hub_custom"


def test_validate_storage_path_rejects_outside_moneyos() -> None:
    with pytest.raises(StoragePolicyError):
        validate_storage_path(r"C:\OtherRoot\cache", root=r"C:\MoneyOS", create=False)


def test_storage_policy_rejects_non_moneyos_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MONEYOS_ROOT", r"C:\Elsewhere")
    with pytest.raises(StoragePolicyError):
        resolve_storage_policy()
