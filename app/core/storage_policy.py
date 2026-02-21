from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
        }


def _resolve_env_path(primary: str, default: str, aliases: tuple[str, ...] = ()) -> StoragePath:
    for key in (primary, *aliases):
        value = os.getenv(key)
        if value:
            return StoragePath(value=str(Path(value).expanduser()), source=f"env:{key}")
    return StoragePath(value=default, source="default")


def _is_windows_drive_path(path: str) -> bool:
    lowered = path.strip().replace("/", "\\").lower()
    return len(lowered) >= 3 and lowered[1:3] == ":\\"


def _validate_not_d_drive(path: str) -> None:
    lowered = path.strip().replace("/", "\\").lower()
    if lowered.startswith("d:\\"):
        raise StoragePolicyError(
            f"StoragePolicy violation: attempted to use {path} (must use C:\\MoneyOS\\...)"
        )


def _validate_c_drive_subpath(path: str, label: str) -> None:
    normalized = path.strip().replace("/", "\\")
    lowered = normalized.lower()
    if lowered.startswith("d:\\"):
        raise StoragePolicyError(
            f"StoragePolicy violation: attempted to use {path} (must use C:\\MoneyOS\\...)"
        )
    if _is_windows_drive_path(normalized) and not lowered.startswith(r"c:\moneyos"):
        raise StoragePolicyError(
            f"StoragePolicy violation: {label}={path} (must use C:\\MoneyOS\\...)"
        )


def resolve_storage_policy() -> StoragePolicy:
    root = _resolve_env_path("MONEYOS_ROOT", r"C:\MoneyOS")
    assets_root = _resolve_env_path("MONEYOS_ASSETS_ROOT", str(Path(root.value) / "assets"))
    output_root = _resolve_env_path("MONEYOS_OUTPUT_ROOT", str(Path(root.value) / "work"))
    cache_root = _resolve_env_path("MONEYOS_CACHE_ROOT", str(Path(root.value) / "cache"))
    temp_root = _resolve_env_path("MONEYOS_TEMP_ROOT", str(Path(root.value) / "tmp"), aliases=("TMPDIR", "TMP", "TEMP"))
    hf_home = _resolve_env_path("HF_HOME", str(Path(cache_root.value) / "huggingface"))
    hf_hub_cache = _resolve_env_path(
        "HF_HUB_CACHE",
        str(Path(hf_home.value) / "hub"),
        aliases=("HUGGINGFACE_HUB_CACHE",),
    )
    hf_datasets_cache = _resolve_env_path("HF_DATASETS_CACHE", str(Path(hf_home.value) / "datasets"))
    torch_home = _resolve_env_path("TORCH_HOME", str(Path(cache_root.value) / "torch"))
    transformers_cache = _resolve_env_path("TRANSFORMERS_CACHE", str(Path(hf_hub_cache.value)))

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
    )
    validate_storage_policy(policy)
    return policy


def validate_storage_policy(policy: StoragePolicy) -> None:
    for item in policy.as_dict().values():
        _validate_not_d_drive(item["path"])
    for key in ("assets_root", "output_root", "cache_root", "hf_home", "hf_hub_cache", "hf_datasets_cache", "torch_home", "temp_root"):
        _validate_c_drive_subpath(policy.as_dict()[key]["path"], key)


def apply_storage_policy_env(policy: StoragePolicy | None = None) -> StoragePolicy:
    policy = policy or resolve_storage_policy()
    os.environ["MONEYOS_ROOT"] = policy.root.value
    os.environ["MONEYOS_ASSETS_ROOT"] = policy.assets_root.value
    os.environ["MONEYOS_OUTPUT_ROOT"] = policy.output_root.value
    os.environ["MONEYOS_CACHE_ROOT"] = policy.cache_root.value
    os.environ["HF_HOME"] = policy.hf_home.value
    os.environ["HF_HUB_CACHE"] = policy.hf_hub_cache.value
    os.environ["HUGGINGFACE_HUB_CACHE"] = policy.hf_hub_cache.value
    os.environ["HF_DATASETS_CACHE"] = policy.hf_datasets_cache.value
    os.environ["TRANSFORMERS_CACHE"] = policy.transformers_cache.value
    os.environ["TORCH_HOME"] = policy.torch_home.value
    os.environ["TMP"] = policy.temp_root.value
    os.environ["TEMP"] = policy.temp_root.value
    os.environ["TMPDIR"] = policy.temp_root.value
    os.environ.setdefault("HUGGINGFACE_HUB_DISABLE_SYMLINKS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    for item in policy.as_dict().values():
        Path(item["path"]).mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = policy.temp_root.value
    return policy


def effective_settings_payload(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    policy = resolve_storage_policy()
    env_keys = [
        "MONEYOS_ROOT",
        "MONEYOS_ASSETS_ROOT",
        "MONEYOS_OUTPUT_ROOT",
        "MONEYOS_CACHE_ROOT",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "HF_DATASETS_CACHE",
        "TRANSFORMERS_CACHE",
        "TORCH_HOME",
        "TMP",
        "TEMP",
        "TMPDIR",
        "PYTORCH_CUDA_ALLOC_CONF",
        "MONEYOS_PYTORCH_ALLOC_CONF",
    ]
    payload = {
        "storage_policy": policy.as_dict(),
        "env": {k: os.getenv(k) for k in env_keys if os.getenv(k) is not None},
    }
    if extra:
        payload.update(extra)
    return payload


def print_effective_settings_banner(extra: dict[str, Any] | None = None, heading: str = "EFFECTIVE SETTINGS") -> dict[str, Any]:
    payload = effective_settings_payload(extra=extra)
    print(f"\n{'=' * 18} {heading} {'=' * 18}")
    for key, value in payload["storage_policy"].items():
        print(f"{key:>18}: {value['path']} [{value['source']}]")
    print("-- env --")
    for key, value in payload["env"].items():
        print(f"{key:>18}={value}")
    if extra:
        print("-- runtime --")
        print(json.dumps(extra, indent=2))
    print("=" * (38 + len(heading)))
    return payload
