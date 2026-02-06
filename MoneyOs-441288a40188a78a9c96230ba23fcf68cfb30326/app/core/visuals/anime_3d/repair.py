from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RepairAction:
    action: str
    applied: bool
    message: str


def repair_failed_render(reason: str) -> RepairAction:
    return RepairAction(action="noop", applied=False, message=f"No repair applied: {reason}")
