from __future__ import annotations


def seconds_to_frames(seconds: float, fps: int) -> int:
    return int(round(seconds * fps))


def ms_to_frames(milliseconds: float, fps: int) -> int:
    return int(round((milliseconds / 1000.0) * fps))
