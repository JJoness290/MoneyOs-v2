from __future__ import annotations

from typing import Any

PHASES = {"proof", "phase2", "phase25", "production"}
_PHASE_ORDER = {
    "proof": 0.0,
    "phase1": 1.0,
    "phase2": 2.0,
    "phase25": 2.5,
    "production": 3.0,
}


def normalize_phase(phase: Any) -> str:
    if phase is None:
        return "phase2"
    if isinstance(phase, (int, float)):
        try:
            numeric = float(phase)
        except (TypeError, ValueError):
            return "phase2"
        if numeric == 2.5 or numeric == 25:
            return "phase25"
        return f"phase{int(numeric)}"
    if isinstance(phase, str):
        raw = phase.strip().lower()
        if not raw:
            return "phase2"
        if raw in PHASES:
            return raw
        compact = raw.replace(" ", "")
        if compact in {"phase2.5", "phase-2.5", "phase25", "2.5", "25"}:
            return "phase25"
        if compact.startswith("phase"):
            suffix = compact[len("phase") :].lstrip("-")
            return _normalize_phase_suffix(suffix)
        return _normalize_phase_suffix(compact)
    return "phase2"


def _normalize_phase_suffix(value: str) -> str:
    if not value:
        return "phase2"
    try:
        numeric = float(value)
    except ValueError:
        return "phase2"
    if numeric == 2.5 or numeric == 25:
        return "phase25"
    return f"phase{int(numeric)}"


def phase_rank(phase: Any) -> float:
    normalized = normalize_phase(phase)
    return _PHASE_ORDER.get(normalized, 2.0)


def is_phase2_or_higher(phase: Any) -> bool:
    return phase_rank(phase) >= 2.0
