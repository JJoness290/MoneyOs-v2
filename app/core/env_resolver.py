from __future__ import annotations

import os


DEFAULT_MONEYOS_ROOT = r"C:\MoneyOS"


def get_env_or_default(name: str, default: str) -> tuple[str, str]:
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return default, "default"
    return str(value), f"env:{name}"
