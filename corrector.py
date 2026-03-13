"""Phase 7 correction loop for schema-invalid records.

This module consumes validator invalid-record artifacts, builds correction prompts
from validation error context, retries LLM-based correction up to a max attempt
count, and writes correction outcomes plus summary statistics.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from dotenv import load_dotenv
from llm_retry import call_with_backoff, maybe_batch_delay
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from validator import SPEC_BY_PREFIX, build_error_payload


load_dotenv(dotenv_path="../.env.local")

DEFAULT_MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_MAX_ATTEMPTS = 3


class CorrectionClient(Protocol):
    """Protocol for generating corrected JSON payloads from prompts."""

    def generate_correction(self, prompt: str) -> str:
        """Return correction response text for the supplied prompt."""


class OpenAICorrectionClient:
    """OpenAI-backed correction client."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found. Add it to mini-projects/.env.local before running corrector.py."
            )

        base_url = os.getenv("OPENAI_BASE_URL")
        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url

        self._client = OpenAI(**kwargs)
        self._model_name = model_name

    def generate_correction(self, prompt: str) -> str:
        """Generate corrected JSON from the model."""
        response = call_with_backoff(
            lambda: self._client.chat.completions.create(
                model=self._model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You repair invalid JSON objects for strict schema validation. "
                            "Return only a corrected JSON object with no markdown."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=1800,
            )
        )
        maybe_batch_delay()
        content = response.choices[0].message.content
        return content or ""


@dataclass
class CorrectionAttempt:
    """One correction attempt with result metadata."""

    attempt_number: int
    succeeded: bool
    error_categories: list[str]
    errors: list[dict[str, Any]]


@dataclass
class CorrectionResult:
    """Outcome for one invalid input record."""

    source_record: dict[str, Any]
    corrected_payload: dict[str, Any] | None
    attempts: list[CorrectionAttempt]

    @property
    def succeeded(self) -> bool:
        return self.corrected_payload is not None


def find_latest_invalid_file(validation_dir: Path) -> Path | None:
    """Find the newest invalid_*.jsonl artifact."""
    matches = sorted(validation_dir.glob("invalid_*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not matches:
        return None
    return matches[-1]


def read_invalid_records(path: Path) -> list[dict[str, Any]]:
    """Read invalid record entries from JSONL."""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract a JSON object from model text response."""
    stripped = text.strip()

    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    first_brace = stripped.find("{")
    last_brace = stripped.rfind("}")
    if first_brace == -1 or last_brace == -1 or last_brace <= first_brace:
        raise json.JSONDecodeError("No JSON object found in model output", stripped, 0)

    candidate = stripped[first_brace : last_brace + 1]
    parsed = json.loads(candidate)
    if not isinstance(parsed, dict):
        raise json.JSONDecodeError("Model output did not contain a JSON object", candidate, 0)
    return parsed


def get_model_for_record_type(record_type: str) -> type[BaseModel]:
    """Resolve pydantic model class from validator record_type."""
    for spec in SPEC_BY_PREFIX.values():
        if spec.output_key == record_type:
            return spec.model
    raise KeyError(f"Unsupported record_type: {record_type}")


def build_correction_prompt(
    *,
    record_type: str,
    errors: list[dict[str, Any]],
    raw_record: Any,
    attempt_number: int,
    max_attempts: int,
) -> str:
    """Build correction prompt from error context + original payload."""
    error_lines: list[str] = []
    for idx, error in enumerate(errors, start=1):
        loc = error.get("loc", "")
        msg = error.get("msg", "")
        etype = error.get("type", "")
        category = error.get("category", "")
        error_lines.append(
            f"{idx}. loc={loc} | type={etype} | category={category} | message={msg}"
        )

    pretty_payload = raw_record if isinstance(raw_record, str) else json.dumps(raw_record, indent=2)

    return (
        f"You are fixing a schema-invalid `{record_type}` record.\n"
        f"Attempt {attempt_number} of {max_attempts}.\n\n"
        "Validation errors to fix:\n"
        f"{os.linesep.join(error_lines) if error_lines else '- unknown error context'}\n\n"
        "Original/last candidate payload:\n"
        f"{pretty_payload}\n\n"
        "Instructions:\n"
        "- Return ONLY one valid JSON object (no markdown, no explanations).\n"
        "- Preserve original intent where possible.\n"
        "- Modify only what is needed to make the object schema-valid.\n"
    )


def run_correction_loop(
    invalid_records: list[dict[str, Any]],
    client: CorrectionClient,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> list[CorrectionResult]:
    """Run correction loop with max-attempt retry policy per invalid record."""
    outcomes: list[CorrectionResult] = []

    for record in invalid_records:
        record_type = str(record.get("record_type", ""))
        model_cls = get_model_for_record_type(record_type)

        current_payload = record.get("raw_record")
        current_errors = list(record.get("errors", []))
        attempts: list[CorrectionAttempt] = []
        corrected: dict[str, Any] | None = None

        for attempt_idx in range(1, max_attempts + 1):
            prompt = build_correction_prompt(
                record_type=record_type,
                errors=current_errors,
                raw_record=current_payload,
                attempt_number=attempt_idx,
                max_attempts=max_attempts,
            )

            response_text = client.generate_correction(prompt)

            try:
                candidate_payload = extract_json_object(response_text)
            except json.JSONDecodeError as exc:
                synthetic_error = {
                    "loc": "__root__",
                    "type": "json_invalid",
                    "msg": str(exc),
                    "category": "format_violations",
                }
                attempts.append(
                    CorrectionAttempt(
                        attempt_number=attempt_idx,
                        succeeded=False,
                        error_categories=["format_violations"],
                        errors=[synthetic_error],
                    )
                )
                current_payload = response_text
                current_errors = [synthetic_error]
                continue

            try:
                model = model_cls.model_validate(candidate_payload)
            except ValidationError as exc:
                categorized_errors, categories = build_error_payload(exc.errors())
                attempts.append(
                    CorrectionAttempt(
                        attempt_number=attempt_idx,
                        succeeded=False,
                        error_categories=sorted(set(categories)),
                        errors=categorized_errors,
                    )
                )
                current_payload = candidate_payload
                current_errors = categorized_errors
                continue

            corrected = model.model_dump(mode="json")
            attempts.append(
                CorrectionAttempt(
                    attempt_number=attempt_idx,
                    succeeded=True,
                    error_categories=[],
                    errors=[],
                )
            )
            break

        outcomes.append(
            CorrectionResult(
                source_record=record,
                corrected_payload=corrected,
                attempts=attempts,
            )
        )

    return outcomes


def compute_correction_report(
    outcomes: list[CorrectionResult],
    max_attempts: int,
    input_source: Path,
) -> dict[str, Any]:
    """Aggregate correction-loop statistics for reporting."""
    total = len(outcomes)
    corrected = sum(1 for outcome in outcomes if outcome.succeeded)
    failed = total - corrected

    attempt_counts = [len(outcome.attempts) for outcome in outcomes]
    avg_attempts = sum(attempt_counts) / len(attempt_counts) if attempt_counts else 0.0

    failure_reason_counts: Counter[str] = Counter()
    for outcome in outcomes:
        if outcome.succeeded:
            continue
        if not outcome.attempts:
            failure_reason_counts["no_attempts"] += 1
            continue
        last_attempt = outcome.attempts[-1]
        if not last_attempt.error_categories:
            failure_reason_counts["unknown"] += 1
            continue
        for category in set(last_attempt.error_categories):
            failure_reason_counts[category] += 1

    success_rate = corrected / total if total else 0.0

    return {
        "input_source": str(input_source),
        "max_attempts": max_attempts,
        "summary": {
            "total_invalid_records": total,
            "corrected_records": corrected,
            "permanently_failed_records": failed,
            "correction_success_rate": round(success_rate, 4),
            "average_attempts": round(avg_attempts, 2),
            "meets_success_criteria_gt_50pct": success_rate > 0.5,
        },
        "attempt_distribution": {
            str(attempt): int(sum(1 for count in attempt_counts if count == attempt))
            for attempt in range(1, max_attempts + 1)
        },
        "common_failure_reasons": dict(failure_reason_counts),
    }


def write_correction_outputs(
    outcomes: list[CorrectionResult],
    report: dict[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    """Write correction artifacts and return written paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    corrected_valid_path = output_dir / f"corrected_valid_{timestamp}.jsonl"
    corrected_invalid_path = output_dir / f"corrected_still_invalid_{timestamp}.jsonl"
    report_path = output_dir / f"correction_report_{timestamp}.json"

    with corrected_valid_path.open("w", encoding="utf-8") as handle:
        for outcome in outcomes:
            if not outcome.succeeded:
                continue
            payload = {
                "record_type": outcome.source_record.get("record_type"),
                "source_file": outcome.source_record.get("source_file"),
                "line_number": outcome.source_record.get("line_number"),
                "attempts_used": len(outcome.attempts),
                "corrected_record": outcome.corrected_payload,
            }
            handle.write(json.dumps(payload, ensure_ascii=True))
            handle.write("\n")

    with corrected_invalid_path.open("w", encoding="utf-8") as handle:
        for outcome in outcomes:
            if outcome.succeeded:
                continue
            payload = {
                "record_type": outcome.source_record.get("record_type"),
                "source_file": outcome.source_record.get("source_file"),
                "line_number": outcome.source_record.get("line_number"),
                "attempts_used": len(outcome.attempts),
                "last_error_categories": outcome.attempts[-1].error_categories if outcome.attempts else [],
                "last_errors": outcome.attempts[-1].errors if outcome.attempts else [],
                "raw_record": outcome.source_record.get("raw_record"),
            }
            handle.write(json.dumps(payload, ensure_ascii=True))
            handle.write("\n")

    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")

    return {
        "corrected_valid": corrected_valid_path,
        "corrected_still_invalid": corrected_invalid_path,
        "report": report_path,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI args for correction loop runs."""
    parser = argparse.ArgumentParser(description="Phase 7 correction loop")
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=Path("outputs/validation"),
        help="Directory containing validator artifacts.",
    )
    parser.add_argument(
        "--invalid-file",
        type=Path,
        default=None,
        help="Optional explicit invalid_*.jsonl path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/corrections"),
        help="Directory to write correction artifacts.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help="Maximum correction attempts per invalid record.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="OpenAI-compatible model name used for correction.",
    )
    return parser.parse_args()


def main() -> None:
    """Run Phase 7 correction loop over latest invalid artifact."""
    args = parse_args()

    invalid_file = args.invalid_file
    if invalid_file is None:
        invalid_file = find_latest_invalid_file(args.validation_dir)

    if invalid_file is None or not invalid_file.exists():
        raise FileNotFoundError(
            f"No invalid artifact found. Looked in {args.validation_dir} for invalid_*.jsonl"
        )

    invalid_records = read_invalid_records(invalid_file)
    client = OpenAICorrectionClient(model_name=args.model_name)
    outcomes = run_correction_loop(
        invalid_records=invalid_records,
        client=client,
        max_attempts=args.max_attempts,
    )
    report = compute_correction_report(
        outcomes=outcomes,
        max_attempts=args.max_attempts,
        input_source=invalid_file,
    )
    paths = write_correction_outputs(
        outcomes=outcomes,
        report=report,
        output_dir=args.output_dir,
    )

    summary = report["summary"]
    print("=" * 72)
    print("PHASE 7 - CORRECTION LOOP")
    print("=" * 72)
    print(f"Input invalid artifact: {invalid_file}")
    print(f"Total invalid records: {summary['total_invalid_records']}")
    print(f"Corrected records: {summary['corrected_records']}")
    print(f"Permanently failed records: {summary['permanently_failed_records']}")
    print(f"Correction success rate: {summary['correction_success_rate']:.2%}")
    print(f"Average attempts: {summary['average_attempts']}")

    print("\nSaved artifacts:")
    for key, path in paths.items():
        print(f"- {key}: {path}")


if __name__ == "__main__":
    main()
