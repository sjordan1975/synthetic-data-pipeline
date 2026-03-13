"""Build pipeline_summary_<timestamp>.json from existing phase artifacts.

This script is a lightweight post-run aggregator (not an orchestrator). It reads
latest artifacts from generation, validation, analysis, and optional correction
outputs, then writes one timestamped pipeline summary JSON.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FAILURE_FLAG_COLUMNS = [
    "experience_mismatch",
    "seniority_mismatch",
    "missing_core_skills",
    "hallucinated_skills",
    "awkward_language",
]


def find_latest_file(directory: Path, pattern: str) -> Path | None:
    """Return the newest file matching pattern, or None if no matches."""
    matches = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not matches:
        return None
    return matches[-1]


def read_json(path: Path) -> dict[str, Any]:
    """Read JSON file as dictionary."""
    return json.loads(path.read_text(encoding="utf-8"))


def count_jsonl_lines(path: Path) -> int:
    """Count non-empty lines in a JSONL file."""
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def summarize_failure_modes(failure_labels_path: Path) -> dict[str, Any]:
    """Aggregate failure flag counts and rates from failure_labels JSONL."""
    total = 0
    counts: Counter[str] = Counter()

    with failure_labels_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            row = json.loads(raw)
            total += 1
            for col in FAILURE_FLAG_COLUMNS:
                if bool(row.get(col)):
                    counts[col] += 1

    return {
        "row_count": total,
        "flag_counts": {col: int(counts.get(col, 0)) for col in FAILURE_FLAG_COLUMNS},
        "flag_rates": {
            col: round(counts.get(col, 0) / total, 4) if total else 0.0
            for col in FAILURE_FLAG_COLUMNS
        },
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI args for summary generation."""
    parser = argparse.ArgumentParser(description="Build pipeline_summary from existing artifacts")
    parser.add_argument(
        "--generated-dir",
        type=Path,
        default=Path("data/generated"),
        help="Directory containing generation artifacts.",
    )
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=Path("outputs/validation"),
        help="Directory containing validation artifacts.",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("outputs/analysis"),
        help="Directory containing analysis artifacts.",
    )
    parser.add_argument(
        "--corrections-dir",
        type=Path,
        default=Path("outputs/corrections"),
        help="Directory containing correction artifacts (optional).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory where pipeline_summary_<timestamp>.json is written.",
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=Path("outputs/metrics"),
        help="Directory containing optional runtime metrics artifacts.",
    )
    parser.add_argument(
        "--stage-times-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON file providing stage durations in seconds. "
            "Expected keys: generation, validation, analysis, correction, visualization"
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Create pipeline summary JSON from latest available artifacts."""
    args = parse_args()

    jobs_path = find_latest_file(args.generated_dir, "jobs_*.jsonl")
    resumes_path = find_latest_file(args.generated_dir, "resumes_*.jsonl")
    pairs_path = find_latest_file(args.generated_dir, "pairs_*.jsonl")
    validated_data_path = find_latest_file(args.validation_dir, "validated_data_*.json")
    failure_labels_path = find_latest_file(args.analysis_dir, "failure_labels_*.jsonl")
    correction_report_path = find_latest_file(args.corrections_dir, "correction_report_*.json")
    api_latency_path = find_latest_file(args.metrics_dir, "api_latency_*.json")

    missing_required: list[str] = []
    if jobs_path is None:
        missing_required.append("jobs_*.jsonl")
    if resumes_path is None:
        missing_required.append("resumes_*.jsonl")
    if pairs_path is None:
        missing_required.append("pairs_*.jsonl")
    if validated_data_path is None:
        missing_required.append("validated_data_*.json")
    if failure_labels_path is None:
        missing_required.append("failure_labels_*.jsonl")

    if missing_required:
        raise FileNotFoundError(
            "Missing required artifacts: " + ", ".join(missing_required)
        )

    validated_payload = read_json(validated_data_path)
    validation_summary = validated_payload.get("summary", {})
    failure_mode_distribution = summarize_failure_modes(failure_labels_path)

    correction_success_rate: float | None = None
    if correction_report_path is not None:
        correction_payload = read_json(correction_report_path)
        correction_success_rate = float(
            correction_payload.get("summary", {}).get("correction_success_rate", 0.0)
        )

    api_latency_summary: dict[str, Any] | None = None
    if api_latency_path is not None:
        api_latency_summary = read_json(api_latency_path)

    stage_times: dict[str, float | None] = {
        "generation": None,
        "validation": None,
        "analysis": None,
        "correction": None,
        "visualization": None,
    }
    if args.stage_times_json is not None and args.stage_times_json.exists():
        provided = read_json(args.stage_times_json)
        for key in stage_times:
            value = provided.get(key)
            stage_times[key] = float(value) if isinstance(value, (int, float)) else None

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / f"pipeline_summary_{timestamp}.json"

    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_artifacts": {
            "jobs": str(jobs_path),
            "resumes": str(resumes_path),
            "pairs": str(pairs_path),
            "validated_data": str(validated_data_path),
            "failure_labels": str(failure_labels_path),
            "correction_report": str(correction_report_path) if correction_report_path else None,
            "api_latency": str(api_latency_path) if api_latency_path else None,
        },
        "summary": {
            "total_records_generated": {
                "jobs": count_jsonl_lines(jobs_path),
                "resumes": count_jsonl_lines(resumes_path),
                "pairs": count_jsonl_lines(pairs_path),
            },
            "validation_success_rate": float(validation_summary.get("validation_success_rate", 0.0)),
            "failure_mode_distribution": failure_mode_distribution,
            "correction_success_rate": correction_success_rate,
            "api_latency": api_latency_summary,
            "processing_time_per_stage_seconds": stage_times,
        },
    }

    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    print("=" * 72)
    print("PIPELINE SUMMARY GENERATED")
    print("=" * 72)
    print(f"Summary file: {summary_path}")


if __name__ == "__main__":
    main()
