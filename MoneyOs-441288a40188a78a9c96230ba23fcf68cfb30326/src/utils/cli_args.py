from __future__ import annotations

from typing import Iterable


def _is_empty(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return False
    text = str(value).strip()
    if not text:
        return True
    lowered = text.lower()
    return lowered in {"none", "null", "auto"}


def add_opt(args: list[str], flag: str, value: object) -> None:
    if _is_empty(value):
        return
    args.extend([flag, str(value)])


def add_flag(args: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        args.append(flag)


def validate_no_empty_value_flags(args: Iterable[str], flags_requiring_value: Iterable[str]) -> None:
    flags = set(flags_requiring_value)
    args_list = list(args)
    for idx, token in enumerate(args_list):
        if token not in flags:
            continue
        next_idx = idx + 1
        if next_idx >= len(args_list):
            raise ValueError(f"Missing value for {token}; command: {' '.join(args_list)}")
        next_value = args_list[next_idx]
        if next_value.startswith("-") or _is_empty(next_value):
            raise ValueError(
                f"Missing value for {token}; got '{next_value}'. Command: {' '.join(args_list)}"
            )
