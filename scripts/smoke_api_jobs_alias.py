"""Smoke check for /api/jobs alias routing.

Usage:
  python scripts/smoke_api_jobs_alias.py

Expected:
  Both requests return HTTP 200 and include a JSON body with `job_id`.

Equivalent curl:
  curl -X POST http://127.0.0.1:8000/api/jobs/anime-episode-60s-3d \
    -H "Content-Type: application/json" --data "{}"
"""

from __future__ import annotations

import json
from urllib.request import Request, urlopen


BASE = "http://127.0.0.1:8000"

REQUIRED_ALIAS_PATHS = (
    '/api/status/test',
    '/api/events/test',
    '/api/videos/test',
    '/api/generate',
    '/api/jobs/anime-episode-60s-3d',
)


def print_alias_targets() -> None:
    print('Expected non-404 alias paths:')
    for path in REQUIRED_ALIAS_PATHS:
        print(f' - {BASE}{path}')


def _post(path: str) -> tuple[int, dict]:
    req = Request(
        f"{BASE}{path}",
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=30) as response:  # nosec B310
        payload = json.loads(response.read().decode("utf-8"))
        return int(response.status), payload


def main() -> None:
    for path in ("/jobs/anime-episode-60s-3d", "/api/jobs/anime-episode-60s-3d"):
        status, payload = _post(path)
        if status != 200 or "job_id" not in payload:
            raise SystemExit(f"FAILED {path}: status={status} payload={payload}")
        print(f"OK {path}: job_id={payload['job_id']}")


if __name__ == "__main__":
    print_alias_targets()
    main()
