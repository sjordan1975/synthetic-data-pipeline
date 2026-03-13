"""Phase 4 schema validation pipeline for generated JSONL artifacts.

This module batch-validates generated jobs/resumes/pairs against Pydantic schema
contracts, separates valid and invalid records, categorizes validation errors,
and writes rubric-aligned validation artifacts.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from schemas.models import JobDescription, Resume, ResumeJobPair


ERROR_MISSING_REQUIRED_FIELDS = "missing_required_fields"
ERROR_TYPE_MISMATCHES = "type_mismatches"
ERROR_FORMAT_VIOLATIONS = "format_violations"
ERROR_LOGICAL_INCONSISTENCIES = "logical_inconsistencies"
ERROR_ENUM_MAPPING_FAILURES = "enum_mapping_failures"

ERROR_CATEGORIES: tuple[str, ...] = (
    ERROR_MISSING_REQUIRED_FIELDS,
    ERROR_TYPE_MISMATCHES,
    ERROR_FORMAT_VIOLATIONS,
    ERROR_LOGICAL_INCONSISTENCIES,
    ERROR_ENUM_MAPPING_FAILURES,
)


@dataclass(frozen=True)
class ModelSpec:
    """Validation model metadata for one generated artifact family."""

    prefix: str
    output_key: str
    model: type[BaseModel]


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(prefix="jobs_", output_key="jobs", model=JobDescription),
    ModelSpec(prefix="resumes_", output_key="resumes", model=Resume),
    ModelSpec(prefix="pairs_", output_key="pairs", model=ResumeJobPair),
)

SPEC_BY_PREFIX: dict[str, ModelSpec] = {spec.prefix: spec for spec in MODEL_SPECS}


@dataclass
class ValidationRunResult:
    """Aggregated result bundle for one validation run."""

    valid_records: dict[str, list[dict[str, Any]]]
    invalid_records: list[dict[str, Any]]
    error_category_counts: Counter[str]
    error_examples_by_category: dict[str, list[dict[str, Any]]]
    total_records: int
    valid_count: int
    invalid_count: int


def to_loc_string(loc: Any) -> str:
    """Render pydantic error location tuples into dotted strings."""
    if isinstance(loc, (tuple, list)):
        return ".".join(str(part) for part in loc)
    return str(loc)


def classify_validation_error(error: dict[str, Any]) -> str:
    """Map a Pydantic error payload to one rubric validation category."""
    error_type = str(error.get("type", ""))
    message = str(error.get("msg", "")).lower()
    loc = to_loc_string(error.get("loc", ())).lower()

    if error_type in {"missing", "missing_argument", "missing_keyword_only_argument"}:
        return ERROR_MISSING_REQUIRED_FIELDS
    if "field required" in message:
        return ERROR_MISSING_REQUIRED_FIELDS

    if error_type == "enum":
        return ERROR_ENUM_MAPPING_FAILURES
    if "input should be" in message and ("'" in message or "enum" in message):
        return ERROR_ENUM_MAPPING_FAILURES

    if error_type in {
        "greater_than",
        "greater_than_equal",
        "less_than",
        "less_than_equal",
    }:
        return ERROR_LOGICAL_INCONSISTENCIES
    if "must be after" in message or "must be greater than" in message:
        return ERROR_LOGICAL_INCONSISTENCIES

    if error_type in {
        "date_parsing",
        "datetime_parsing",
        "time_parsing",
        "string_pattern_mismatch",
        "json_invalid",
    }:
        return ERROR_FORMAT_VIOLATIONS
    if "invalid phone number format" in message:
        return ERROR_FORMAT_VIOLATIONS
    if "valid email address" in message:
        return ERROR_FORMAT_VIOLATIONS
    if "format" in message and any(token in loc for token in ("email", "phone", "date")):
        return ERROR_FORMAT_VIOLATIONS

    if error_type.endswith("_type"):
        return ERROR_TYPE_MISMATCHES
    if "input should be a valid" in message:
        return ERROR_TYPE_MISMATCHES

    # Keep categorization exhaustive for rubric reporting.
    return ERROR_TYPE_MISMATCHES


def build_error_payload(errors: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Attach rubric categories to raw error objects."""
    payload: list[dict[str, Any]] = []
    categories: list[str] = []

    for error in errors:
        category = classify_validation_error(error)
        categories.append(category)
        payload.append(
            {
                "loc": to_loc_string(error.get("loc", ())),
                "type": error.get("type"),
                "msg": error.get("msg"),
                "category": category,
            }
        )

    return payload, categories


def iter_jsonl_records(file_path: Path) -> list[tuple[int, str]]:
    """Read non-empty JSONL lines with line numbers."""
    records: list[tuple[int, str]] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            records.append((line_number, line))
    return records


def validate_file(
    file_path: Path,
    spec: ModelSpec,
    valid_records: dict[str, list[dict[str, Any]]],
    invalid_records: list[dict[str, Any]],
    error_category_counts: Counter[str],
    error_examples_by_category: dict[str, list[dict[str, Any]]],
) -> tuple[int, int]:
    """Validate one JSONL file and append into aggregate collectors."""
    file_total = 0
    file_valid = 0

    for line_number, raw_line in iter_jsonl_records(file_path):
        file_total += 1

        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            synthetic_error = {
                "loc": ("__root__",),
                "type": "json_invalid",
                "msg": str(exc),
            }
            categorized_errors, categories = build_error_payload([synthetic_error])
            error_category_counts.update(categories)
            error_examples_by_category[categories[0]].append(
                {
                    "source_file": str(file_path),
                    "line_number": line_number,
                    "loc": "__root__",
                    "msg": str(exc),
                    "type": "json_invalid",
                }
            )
            invalid_records.append(
                {
                    "source_file": str(file_path),
                    "record_type": spec.output_key,
                    "line_number": line_number,
                    "error_count": 1,
                    "error_categories": sorted(set(categories)),
                    "errors": categorized_errors,
                    "raw_record": raw_line,
                }
            )
            continue

        try:
            model = spec.model.model_validate(payload)
        except ValidationError as exc:
            raw_errors = exc.errors()
            categorized_errors, categories = build_error_payload(raw_errors)
            error_category_counts.update(categories)
            for categorized in categorized_errors:
                if len(error_examples_by_category[categorized["category"]]) >= 5:
                    continue
                error_examples_by_category[categorized["category"]].append(
                    {
                        "source_file": str(file_path),
                        "line_number": line_number,
                        "loc": categorized["loc"],
                        "msg": categorized["msg"],
                        "type": categorized["type"],
                    }
                )

            invalid_records.append(
                {
                    "source_file": str(file_path),
                    "record_type": spec.output_key,
                    "line_number": line_number,
                    "error_count": len(categorized_errors),
                    "error_categories": sorted(set(categories)),
                    "errors": categorized_errors,
                    "raw_record": payload,
                }
            )
            continue

        valid_records[spec.output_key].append(model.model_dump(mode="json"))
        file_valid += 1

    return file_total, file_valid


def discover_input_files(input_dir: Path) -> list[tuple[Path, ModelSpec]]:
    """Find generated JSONL files that match known artifact prefixes."""
    discovered: list[tuple[Path, ModelSpec]] = []

    for file_path in sorted(input_dir.glob("*.jsonl")):
        for prefix, spec in SPEC_BY_PREFIX.items():
            if file_path.name.startswith(prefix):
                discovered.append((file_path, spec))
                break

    return discovered


def run_validation(input_dir: Path) -> ValidationRunResult:
    """Execute schema validation over discovered generated artifacts."""
    files = discover_input_files(input_dir)
    if not files:
        raise FileNotFoundError(
            f"No generated JSONL files found in {input_dir}. Expected prefixes: "
            f"{', '.join(spec.prefix for spec in MODEL_SPECS)}"
        )

    valid_records: dict[str, list[dict[str, Any]]] = {spec.output_key: [] for spec in MODEL_SPECS}
    invalid_records: list[dict[str, Any]] = []
    error_category_counts: Counter[str] = Counter({category: 0 for category in ERROR_CATEGORIES})
    error_examples_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)

    total_records = 0
    valid_count = 0

    for file_path, spec in files:
        file_total, file_valid = validate_file(
            file_path=file_path,
            spec=spec,
            valid_records=valid_records,
            invalid_records=invalid_records,
            error_category_counts=error_category_counts,
            error_examples_by_category=error_examples_by_category,
        )
        total_records += file_total
        valid_count += file_valid

    invalid_count = total_records - valid_count

    # Keep all known categories visible even when count is zero.
    for category in ERROR_CATEGORIES:
        error_examples_by_category.setdefault(category, [])

    return ValidationRunResult(
        valid_records=valid_records,
        invalid_records=invalid_records,
        error_category_counts=error_category_counts,
        error_examples_by_category=dict(error_examples_by_category),
        total_records=total_records,
        valid_count=valid_count,
        invalid_count=invalid_count,
    )


def write_outputs(result: ValidationRunResult, output_dir: Path) -> dict[str, Path]:
    """Write rubric output artifacts and return generated file paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    validated_path = output_dir / f"validated_data_{timestamp}.json"
    invalid_path = output_dir / f"invalid_{timestamp}.jsonl"
    failure_modes_path = output_dir / f"schema_failure_modes_{timestamp}.json"

    success_rate = (
        result.valid_count / result.total_records if result.total_records else 0.0
    )

    validated_payload = {
        "summary": {
            "total_records": result.total_records,
            "valid_records": result.valid_count,
            "invalid_records": result.invalid_count,
            "validation_success_rate": round(success_rate, 4),
        },
        "records": result.valid_records,
    }
    validated_path.write_text(
        json.dumps(validated_payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    with invalid_path.open("w", encoding="utf-8") as handle:
        for record in result.invalid_records:
            handle.write(json.dumps(record, ensure_ascii=True))
            handle.write("\n")

    failure_modes_payload = {
        "summary": {
            "total_records": result.total_records,
            "valid_records": result.valid_count,
            "invalid_records": result.invalid_count,
            "validation_success_rate": round(success_rate, 4),
        },
        "error_type_distribution": {
            category: int(result.error_category_counts.get(category, 0))
            for category in ERROR_CATEGORIES
        },
        "error_examples": result.error_examples_by_category,
    }
    failure_modes_path.write_text(
        json.dumps(failure_modes_payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    return {
        "validated": validated_path,
        "invalid": invalid_path,
        "failure_modes": failure_modes_path,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for validation run."""
    parser = argparse.ArgumentParser(description="Phase 4 schema validator")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/generated"),
        help="Directory containing generated JSONL files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/validation"),
        help="Directory to write validator artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry-point for Phase 4 schema validation."""
    args = parse_args()
    result = run_validation(input_dir=args.input_dir)
    paths = write_outputs(result=result, output_dir=args.output_dir)

    success_rate = (
        result.valid_count / result.total_records if result.total_records else 0.0
    )

    print("=" * 72)
    print("PHASE 4 - SCHEMA VALIDATION PIPELINE")
    print("=" * 72)
    print(f"Input directory: {args.input_dir}")
    print(f"Total records: {result.total_records}")
    print(f"Valid records: {result.valid_count}")
    print(f"Invalid records: {result.invalid_count}")
    print(f"Validation success rate: {success_rate:.2%}")

    print("\nError type distribution:")
    for category in ERROR_CATEGORIES:
        print(f"- {category}: {result.error_category_counts.get(category, 0)}")

    print("\nSaved artifacts:")
    print(f"- {paths['validated']}")
    print(f"- {paths['invalid']}")
    print(f"- {paths['failure_modes']}")


if __name__ == "__main__":
    main()
