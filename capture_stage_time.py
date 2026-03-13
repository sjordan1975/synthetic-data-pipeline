"""Capture runtime for one pipeline stage and append/update a stage-times JSON file.

This is intentionally not a full pipeline orchestrator. Run it per phase, e.g.:

python capture_stage_time.py --stage generation --times-json outputs/stage_times.json -- \
  python generator.py --num-jobs 50 --resumes-per-job 5
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

VALID_STAGES = {"generation", "validation", "analysis", "correction", "visualization", "api_benchmark"}


def read_json(path: Path) -> dict[str, Any]:
    """Read JSON dictionary from file."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON dictionary to file with indentation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Capture elapsed time for one pipeline stage")
    parser.add_argument("--stage", required=True, help="Stage name to record")
    parser.add_argument(
        "--times-json",
        type=Path,
        default=Path("outputs/stage_times.json"),
        help="Path for the stage-times JSON artifact.",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to run after '--'. Example: -- python validator.py",
    )
    return parser.parse_args()


def main() -> None:
    """Run command and persist elapsed seconds for provided stage."""
    args = parse_args()
    stage = args.stage.strip().lower()
    if stage not in VALID_STAGES:
        raise ValueError(f"Invalid --stage '{args.stage}'. Valid values: {sorted(VALID_STAGES)}")

    if not args.command:
        raise ValueError("No command provided. Use '--' before the command to run.")

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("No command provided after '--'.")

    started_at = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    completed = subprocess.run(command, check=False)
    elapsed_s = round(time.perf_counter() - t0, 4)
    ended_at = datetime.now(timezone.utc)

    payload: dict[str, Any] = {}
    if args.times_json.exists():
        payload = read_json(args.times_json)

    durations = payload.get("durations_seconds", {})
    durations[stage] = elapsed_s
    payload["durations_seconds"] = durations

    metadata = payload.get("stage_metadata", {})
    metadata[stage] = {
        "command": command,
        "exit_code": completed.returncode,
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
    }
    payload["stage_metadata"] = metadata
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()

    write_json(args.times_json, payload)

    print("=" * 72)
    print("STAGE TIME CAPTURED")
    print("=" * 72)
    print(f"stage={stage} elapsed_s={elapsed_s} exit_code={completed.returncode}")
    print(f"times_json={args.times_json}")

    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
