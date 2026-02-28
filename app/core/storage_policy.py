from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.env_resolver import DEFAULT_MONEYOS_ROOT, get_env_or_default


class StoragePolicyError(RuntimeError):
    pass


@dataclass(frozen=True)
class StoragePath:
    value: str
    source: str


@dataclass(frozen=True)
class StoragePolicy:
    root: StoragePath
    assets_root: StoragePath
    output_root: StoragePath
    cache_root: StoragePath
    hf_home: StoragePath
    hf_hub_cache: StoragePath
    hf_datasets_cache: StoragePath
    torch_home: StoragePath
    transformers_cache: StoragePath
    temp_root: StoragePath
    logs_root: StoragePath

    def as_dict(self) -> dict[str, dict[str, str]]:
        return {
            "root": {"path": self.root.value, "source": self.root.source},
            "assets_root": {"path": self.assets_root.value, "source": self.assets_root.source},
            "output_root": {"path": self.output_root.value, "source": self.output_root.source},
            "cache_root": {"path": self.cache_root.value, "source": self.cache_root.source},
            "hf_home": {"path": self.hf_home.value, "source": self.hf_home.source},
            "hf_hub_cache": {"path": self.hf_hub_cache.value, "source": self.hf_hub_cache.source},
            "hf_datasets_cache": {"path": self.hf_datasets_cache.value, "source": self.hf_datasets_cache.source},
            "torch_home": {"path": self.torch_home.value, "source": self.torch_home.source},
            "transformers_cache": {"path": self.transformers_cache.value, "source": self.transformers_cache.source},
            "temp_root": {"path": self.temp_root.value, "source": self.temp_root.source},
            "logs_root": {"path": self.logs_root.value, "source": self.logs_root.source},
        }


def _normalize_path(path: str) -> str:
    return str(path).strip().replace("/", "\\")


def _is_windows_drive_path(path: str) -> bool:
    value = _normalize_path(path)
    return len(value) >= 3 and value[1:3] == ":\\"


def resolve_moneyos_root() -> StoragePath:
    value, source = get_env_or_default("MONEYOS_ROOT", DEFAULT_MONEYOS_ROOT)
    return StoragePath(value=_normalize_path(value), source=source)


def _resolve_env_path(name: str, default: str) -> StoragePath:
    value, source = get_env_or_default(name, default)
    return StoragePath(value=_normalize_path(value), source=source)


def validate_storage_path(path: str, *, root: str = DEFAULT_MONEYOS_ROOT, create: bool = True) -> Path:
    normalized = _normalize_path(path)
    lowered = normalized.lower()
    root_lower = _normalize_path(root).lower()
    if lowered.startswith("d:\\"):
        raise StoragePolicyError(
            f"StoragePolicy violation: attempted to use {normalized} (must use C:\\MoneyOS\\...)"
        )
    if _is_windows_drive_path(normalized) and not lowered.startswith(root_lower):
        raise StoragePolicyError(
            f"StoragePolicy violation: attempted to use {normalized} (must use {root}\\...)"
        )
    output = Path(normalized)
    if create:
        output.mkdir(parents=True, exist_ok=True)
    return output


def _resolve_hf_hub_cache(root_cache_hf: str) -> StoragePath:
    # Required priority: HF_HUB_CACHE > HF_HOME > HUGGINGFACE_HUB_CACHE > HF_DATASETS_CACHE > TORCH_HOME > XDG_CACHE_HOME > fallback
    for key in ("HF_HUB_CACHE", "HF_HOME", "HUGGINGFACE_HUB_CACHE", "HF_DATASETS_CACHE", "TORCH_HOME", "XDG_CACHE_HOME"):
        raw = os.getenv(key)
        if not raw:
            continue
        base = _normalize_path(raw)
        if key == "HF_HUB_CACHE":
            return StoragePath(value=base, source=f"env:{key}")
        if key in {"HF_HOME", "HUGGINGFACE_HUB_CACHE", "XDG_CACHE_HOME"}:
            return StoragePath(value=_normalize_path(str(Path(base) / "hub")), source=f"env:{key}")
        return StoragePath(value=base, source=f"env:{key}")
    return StoragePath(value=_normalize_path(str(Path(root_cache_hf) / "hub")), source="default")


def resolve_storage_policy() -> StoragePolicy:
    root = resolve_moneyos_root()
    assets_root = _resolve_env_path("MONEYOS_ASSETS_ROOT", str(Path(root.value) / "assets"))
    output_root = _resolve_env_path("MONEYOS_OUTPUT_ROOT", str(Path(root.value) / "work"))
    cache_root = _resolve_env_path("MONEYOS_CACHE_ROOT", str(Path(root.value) / "cache"))
    logs_root = _resolve_env_path("MONEYOS_LOGS_ROOT", str(Path(root.value) / "logs"))

    hf_home = _resolve_env_path("HF_HOME", str(Path(cache_root.value) / "huggingface"))
    hf_hub_cache = _resolve_hf_hub_cache(hf_home.value)
    hf_datasets_cache = _resolve_env_path("HF_DATASETS_CACHE", str(Path(hf_home.value) / "datasets"))
    torch_home = _resolve_env_path("TORCH_HOME", str(Path(cache_root.value) / "torch"))
    transformers_cache = _resolve_env_path("TRANSFORMERS_CACHE", hf_hub_cache.value)
    temp_root = _resolve_env_path("MONEYOS_TEMP_ROOT", str(Path(root.value) / "tmp"))

    policy = StoragePolicy(
        root=root,
        assets_root=assets_root,
        output_root=output_root,
        cache_root=cache_root,
        hf_home=hf_home,
        hf_hub_cache=hf_hub_cache,
        hf_datasets_cache=hf_datasets_cache,
        torch_home=torch_home,
        transformers_cache=transformers_cache,
        temp_root=temp_root,
        logs_root=logs_root,
    )
    validate_storage_policy(policy)
    return policy


def validate_storage_policy(policy: StoragePolicy) -> None:
    root = policy.root.value
    if _normalize_path(root).lower() != DEFAULT_MONEYOS_ROOT.lower():
        raise StoragePolicyError(
            f"StoragePolicy violation: MONEYOS_ROOT={root} (must be {DEFAULT_MONEYOS_ROOT})"
        )
    for item in policy.as_dict().values():
        validate_storage_path(item["path"], root=root, create=False)


def apply_storage_policy_env(policy: StoragePolicy | None = None) -> StoragePolicy:
    policy = policy or resolve_storage_policy()
    os.environ.setdefault("MONEYOS_ROOT", policy.root.value)
    os.environ.setdefault("MONEYOS_ASSETS_ROOT", policy.assets_root.value)
    os.environ.setdefault("MONEYOS_OUTPUT_ROOT", policy.output_root.value)
    os.environ.setdefault("MONEYOS_CACHE_ROOT", policy.cache_root.value)
    os.environ.setdefault("MONEYOS_LOGS_ROOT", policy.logs_root.value)
    os.environ.setdefault("HF_HOME", policy.hf_home.value)
    os.environ.setdefault("HF_HUB_CACHE", policy.hf_hub_cache.value)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", policy.hf_hub_cache.value)
    os.environ.setdefault("HF_DATASETS_CACHE", policy.hf_datasets_cache.value)
    os.environ.setdefault("TRANSFORMERS_CACHE", policy.transformers_cache.value)
    os.environ.setdefault("TORCH_HOME", policy.torch_home.value)
    os.environ.setdefault("TEMP", policy.temp_root.value)
    os.environ.setdefault("TMP", policy.temp_root.value)
    os.environ.setdefault("TMPDIR", policy.temp_root.value)
    os.environ.setdefault("HUGGINGFACE_HUB_DISABLE_SYMLINKS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    for item in policy.as_dict().values():
        validate_storage_path(item["path"], root=policy.root.value, create=True)
    tempfile.tempdir = policy.temp_root.value
    return policy


def effective_settings_payload(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    policy = resolve_storage_policy()
    env_keys = [
        "MONEYOS_ROOT",
        "MONEYOS_ASSETS_ROOT",
        "MONEYOS_OUTPUT_ROOT",
        "MONEYOS_CACHE_ROOT",
        "MONEYOS_LOGS_ROOT",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "HF_DATASETS_CACHE",
        "TORCH_HOME",
        "TRANSFORMERS_CACHE",
        "TEMP",
        "TMP",
        "TMPDIR",
        "PYTORCH_CUDA_ALLOC_CONF",
        "MONEYOS_PYTORCH_ALLOC_CONF",
        "MONEYOS_VRAM_FRACTION",
        "MONEYOS_VRAM_FRACTION_LOCK",
        "MONEYOS_VRAM_FRACTION_EFFECTIVE",
        "MONEYOS_MAX_CONCURRENCY",
    ]
    payload = {
        "storage_policy": policy.as_dict(),
        "env": {k: os.getenv(k) for k in env_keys if os.getenv(k) is not None},
    }
    if extra:
        payload.update(extra)
    return payload


def print_effective_settings_banner(extra: dict[str, Any] | None = None, heading: str = "MONEYOS EFFECTIVE SETTINGS") -> dict[str, Any]:
    payload = effective_settings_payload(extra=extra)
    print("=== MONEYOS EFFECTIVE SETTINGS ===")
    print(f"heading={heading}")
    for key, value in payload["storage_policy"].items():
        print(f"{key:>18}: {value['path']} [{value['source']}]")
    print("-- env --")
    for key, value in payload["env"].items():
        print(f"{key:>18}={value}")
    if extra:
        print("-- runtime --")
        print(json.dumps(extra, indent=2))
    print("=== /MONEYOS EFFECTIVE SETTINGS ===")
    return payload
