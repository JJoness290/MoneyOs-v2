from __future__ import annotations

import subprocess
from typing import Iterable


def estimate_windows_cmd_length(args: Iterable[str]) -> int:
    return len(subprocess.list2cmdline(list(args)))
