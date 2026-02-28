from __future__ import annotations

import re

_LABEL_PREFIXES = (
    "scene",
    "narration:",
    "narrator:",
    "sfx:",
    "music:",
    "cut to",
    "int.",
    "ext.",
)


def sanitize_spoken_script(text: str) -> str:
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        lowered = line.lower()
        if lowered.startswith(_LABEL_PREFIXES):
            continue
        line = re.sub(r"\[[^\]]*\]", " ", line)
        line = re.sub(r"\([^\)]*\)", " ", line)
        line = re.sub(r"^\s*[A-Z][A-Z0-9_\- ]{1,30}:\s*", "", line)
        line = line.replace("Narration:", "").replace("NARRATION:", "")
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)
