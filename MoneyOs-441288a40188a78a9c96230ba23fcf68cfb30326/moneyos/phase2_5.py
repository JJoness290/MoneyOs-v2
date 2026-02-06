from __future__ import annotations

import argparse
import os
import uuid

from app.main import _run_hybrid_episode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Phase 2.5 hybrid episode render.")
    parser.add_argument("--out", dest="output_root", help="Output root (e.g. C:\\MoneyOS\\work\\p2)")
    parser.add_argument("--seconds", type=float, default=30.0, help="Target duration in seconds.")
    parser.add_argument("--episode-id", dest="episode_id", help="Optional episode/job id override.")
    args = parser.parse_args()

    if args.output_root:
        os.environ["MONEYOS_OUTPUT_ROOT"] = args.output_root
    os.environ.setdefault("MONEYOS_PHASE", "phase25")
    os.environ["MONEYOS_TARGET_SECONDS"] = f"{args.seconds:.3f}"

    job_id = args.episode_id or uuid.uuid4().hex
    print(f"[CLI] phase25 episode_id={job_id} target_seconds={args.seconds:.2f}")
    _run_hybrid_episode(job_id, target_seconds=args.seconds)


if __name__ == "__main__":
    main()
