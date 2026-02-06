from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

try:
    import ctypes
except Exception:  # noqa: BLE001
    ctypes = None


def get_short_workdir() -> Path:
    root = os.getenv("MONEYOS_SHORT_WORKDIR", r"C:\MoneyOS\work")
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def shorten_component(text: str) -> str:
    slug = "".join(ch for ch in text if ch.isalnum()).lower()[:12] or "item"
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{slug}_{digest}"


def ensure_max_path(path: Path, max_len: int) -> Path:
    if len(str(path)) <= max_len:
        return path
    parts = list(path.parts)
    shortened = [parts[0]]
    for part in parts[1:]:
        shortened.append(shorten_component(part))
        candidate = Path(*shortened)
        if len(str(candidate)) > max_len:
            continue
    return Path(*shortened)


def safe_join(*parts: str, max_len: int | None = None) -> Path:
    base = get_short_workdir()
    path = base.joinpath(*parts)
    max_len = max_len or int(os.getenv("MONEYOS_PATH_MAX", "220"))
    return ensure_max_path(path, max_len)


def to_win_extended_path(path: Path) -> str:
    value = str(path.resolve())
    if value.startswith("\\\\?\\"):
        return value
    return f"\\\\?\\{value}"


def get_8dot3_short_path(path: Path) -> str:
    value = str(path)
    if ctypes is None:
        return value
    buf = ctypes.create_unicode_buffer(260)
    try:
        result = ctypes.windll.kernel32.GetShortPathNameW(value, buf, 260)
    except Exception:  # noqa: BLE001
        return value
    if result == 0:
        return value
    return buf.value or value


def record_path_mapping(long_path: Path, short_path: Path, mapping_json: Path) -> None:
    mapping_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    if mapping_json.exists():
        try:
            payload = json.loads(mapping_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
    payload[str(long_path)] = str(short_path)
    mapping_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def planned_paths_preflight(paths: Iterable[Path], max_len: int | None = None) -> tuple[bool, Path, int]:
    max_len = max_len or int(os.getenv("MONEYOS_PATH_MAX", "220"))
    longest = max(paths, key=lambda p: len(str(p)))
    longest_len = len(str(longest))
    return longest_len <= max_len, longest, longest_len
