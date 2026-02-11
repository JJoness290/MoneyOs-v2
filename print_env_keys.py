import re
from pathlib import Path

p = Path("app/core/visuals/ai_video/backends/cogvideox.py")
s = p.read_text(encoding="utf-8", errors="ignore")

# Find all getenv("MONEYOS_...") keys
keys = sorted(set(re.findall(r'os\.getenv\("([^"]+)"', s)))
keys = [k for k in keys if k.startswith("MONEYOS_")]

print("\n".join(keys))
