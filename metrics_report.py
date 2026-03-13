"""Generate a rubric-ready markdown report from runtime metric artifacts.

This script reads the latest available metrics artifacts and emits a concise
markdown summary suitable for README/submission evidence.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def find_latest_file(directory: Path, pattern: str) -> Path | None:
    """Return newest file matching pattern in directory, if any."""
    matches = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not matches:
        return None
    return matches[-1]


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file into a dictionary."""
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_seconds(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.3f}s"
    return "n/a"


def _fmt_percent(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value * 100:.1f}%"
    return "n/a"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Generate markdown metrics report")
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=Path("outputs/metrics"),
        help="Directory containing api_latency_*.json and report output.",
    )
    parser.add_argument(
        "--pipeline-summary-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory containing pipeline_summary_*.json.",
    )
    parser.add_argument(
        "--stage-times-json",
        type=Path,
        default=Path("outputs/stage_times.json"),
        help="Optional stage timing artifact JSON.",
    )
    parser.add_argument(
        "--correction-target",
        type=float,
        default=0.5,
        help="Target correction success rate used for pass/fail statement.",
    )
    return parser.parse_args()


def _extract_api_rows(api_latency_payload: dict[str, Any]) -> list[str]:
    rows: list[str] = []
    rules_p95 = (
        api_latency_payload.get("rules_only", {})
        .get("client", {})
        .get("p95_s")
    )
    judge_p95 = (
        api_latency_payload.get("with_judge", {})
        .get("client", {})
        .get("p95_s")
    )

    rules_target = (
        api_latency_payload.get("rules_only", {})
        .get("client", {})
        .get("target_s", 2.0)
    )
    judge_target = (
        api_latency_payload.get("with_judge", {})
        .get("client", {})
        .get("target_s", 10.0)
    )

    if isinstance(rules_p95, (int, float)):
        rules_status = "PASS" if rules_p95 <= float(rules_target) else "FAIL"
    else:
        rules_status = "NOT RUN"

    if isinstance(judge_p95, (int, float)):
        judge_status = "PASS" if judge_p95 <= float(judge_target) else "FAIL"
    else:
        judge_status = "NOT RUN"

    rows.append(
        f"- Rules-only p95 latency: {_fmt_seconds(rules_p95)} vs target {_fmt_seconds(rules_target)} -> {rules_status}"
    )
    rows.append(
        f"- With-judge p95 latency: {_fmt_seconds(judge_p95)} vs target {_fmt_seconds(judge_target)} -> {judge_status}"
    )
    return rows


def main() -> None:
    """Build markdown metrics report from latest available artifacts."""
    args = parse_args()

    api_latency_path = find_latest_file(args.metrics_dir, "api_latency_*.json")
    pipeline_summary_path = find_latest_file(args.pipeline_summary_dir, "pipeline_summary_*.json")

    api_latency_payload: dict[str, Any] | None = None
    if api_latency_path is not None:
        api_latency_payload = read_json(api_latency_path)

    pipeline_summary_payload: dict[str, Any] | None = None
    if pipeline_summary_path is not None:
        pipeline_summary_payload = read_json(pipeline_summary_path)

    stage_times_payload: dict[str, Any] | None = None
    if args.stage_times_json.exists():
        stage_times_payload = read_json(args.stage_times_json)

    correction_rate = None
    if pipeline_summary_payload is not None:
        correction_rate = (
            pipeline_summary_payload.get("summary", {})
            .get("correction_success_rate")
        )

    has_correction_rate = isinstance(correction_rate, (int, float))
    correction_pass = has_correction_rate and correction_rate >= args.correction_target
    if not has_correction_rate:
        correction_status = "NOT RUN"
    else:
        correction_status = "PASS" if correction_pass else "FAIL"

    lines: list[str] = []
    lines.append("# Metrics Report")
    lines.append("")
    lines.append(f"Generated at: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append("## Evidence Sources")
    lines.append(f"- API latency artifact: `{api_latency_path}`" if api_latency_path else "- API latency artifact: not found")
    lines.append(
        f"- Pipeline summary artifact: `{pipeline_summary_path}`"
        if pipeline_summary_path
        else "- Pipeline summary artifact: not found"
    )
    lines.append(
        f"- Stage times artifact: `{args.stage_times_json}`"
        if stage_times_payload is not None
        else "- Stage times artifact: not found"
    )
    lines.append("")

    lines.append("## API Latency Targets")
    if api_latency_payload is None:
        lines.append("- No API latency artifact found.")
    else:
        lines.extend(_extract_api_rows(api_latency_payload))
    lines.append("")

    lines.append("## Correction Effectiveness")
    lines.append(
        f"- Correction success rate: {_fmt_percent(correction_rate)} vs target {_fmt_percent(args.correction_target)} -> {correction_status}"
    )
    lines.append("")

    lines.append("## Stage Runtimes")
    stage_durations = {}
    if stage_times_payload is not None:
        stage_durations = stage_times_payload.get("durations_seconds", {})

    if not stage_durations:
        lines.append("- No stage runtime data found.")
    else:
        for stage in ["generation", "validation", "analysis", "correction", "visualization", "api_benchmark"]:
            lines.append(f"- {stage}: {_fmt_seconds(stage_durations.get(stage))}")

    lines.append("")
    lines.append("## Submission-Ready Summary")
    if api_latency_payload is not None:
        rules_p95 = api_latency_payload.get("rules_only", {}).get("client", {}).get("p95_s")
        judge_p95 = api_latency_payload.get("with_judge", {}).get("client", {}).get("p95_s")
        lines.append(
            "- API response latency evidence: "
            f"rules-only p95={_fmt_seconds(rules_p95)}, with-judge p95={_fmt_seconds(judge_p95)}."
        )
    else:
        lines.append("- API response latency evidence: not available.")

    if has_correction_rate:
        lines.append(
            "- Correction loop evidence: "
            f"success rate={_fmt_percent(correction_rate)} against target {_fmt_percent(args.correction_target)} ({correction_status})."
        )
    else:
        lines.append(
            "- Correction loop evidence: not run or not available in pipeline summary."
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    args.metrics_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.metrics_dir / f"metrics_report_{timestamp}.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=" * 72)
    print("METRICS REPORT GENERATED")
    print("=" * 72)
    print(f"Report file: {report_path}")


if __name__ == "__main__":
    main()
