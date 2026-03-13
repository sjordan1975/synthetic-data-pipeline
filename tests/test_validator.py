"""Targeted tests for the Phase 4 validator pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import validator
from schemas.models import JobDescription, ResumeJobPair


def build_valid_job_description(valid_job_requirements) -> JobDescription:
    """Build a valid JobDescription for validator integration tests."""
    return JobDescription(
        company="StartupXYZ",
        industry="Software",
        company_size="51-200",
        position="Software Engineer",
        location="San Francisco, CA",
        description="We are looking for an experienced engineer to build APIs and internal tools.",
        requirements=valid_job_requirements,
        metadata={
            "trace_id": "job-test-001",
            "generated_at": "2024-01-15T10:00:00Z",
            "is_niche_role": False,
            "used_fallback": False,
        },
    )


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            {"type": "missing", "msg": "Field required", "loc": ("resume", "skills")},
            validator.ERROR_MISSING_REQUIRED_FIELDS,
        ),
        (
            {
                "type": "enum",
                "msg": "Input should be 'good' or 'poor'",
                "loc": ("fit_level",),
            },
            validator.ERROR_ENUM_MAPPING_FAILURES,
        ),
        (
            {
                "type": "greater_than_equal",
                "msg": "Input should be greater than or equal to 0",
                "loc": ("requirements", "years_experience"),
            },
            validator.ERROR_LOGICAL_INCONSISTENCIES,
        ),
        (
            {
                "type": "json_invalid",
                "msg": "Expecting value",
                "loc": ("__root__",),
            },
            validator.ERROR_FORMAT_VIOLATIONS,
        ),
        (
            {
                "type": "int_type",
                "msg": "Input should be a valid integer",
                "loc": ("requirements", "years_experience"),
            },
            validator.ERROR_TYPE_MISMATCHES,
        ),
    ],
)
def test_classify_validation_error_maps_to_rubric_category(error: dict, expected: str) -> None:
    """Validation errors should map into one rubric category."""
    assert validator.classify_validation_error(error) == expected


def test_run_validation_collects_valid_and_invalid_records(
    tmp_path: Path,
    valid_resume,
    valid_job_requirements,
) -> None:
    """Validator should separate valid/invalid records and track category counts."""
    job = build_valid_job_description(valid_job_requirements)
    pair = ResumeJobPair(
        resume=valid_resume,
        job_description=job,
        fit_level="good",
        trace_id="pair-test-001",
    )

    jobs_file = tmp_path / "jobs_20260101_000000.jsonl"
    resumes_file = tmp_path / "resumes_20260101_000000.jsonl"
    pairs_file = tmp_path / "pairs_20260101_000000.jsonl"

    jobs_file.write_text(
        "\n".join(
            [
                job.model_dump_json(),
                "{ this is invalid json",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    invalid_resume_payload = valid_resume.model_dump(mode="json")
    invalid_resume_payload.pop("contact_info")
    resumes_file.write_text(
        json.dumps(invalid_resume_payload) + "\n",
        encoding="utf-8",
    )

    pairs_file.write_text(pair.model_dump_json() + "\n", encoding="utf-8")

    result = validator.run_validation(input_dir=tmp_path)

    assert result.total_records == 4
    assert result.valid_count == 2
    assert result.invalid_count == 2
    assert len(result.invalid_records) == 2
    assert result.error_category_counts[validator.ERROR_FORMAT_VIOLATIONS] >= 1
    assert result.error_category_counts[validator.ERROR_MISSING_REQUIRED_FIELDS] >= 1


def test_write_outputs_creates_expected_artifacts(
    tmp_path: Path,
    valid_resume,
    valid_job_requirements,
) -> None:
    """Validator output writer should emit all rubric output files."""
    job = build_valid_job_description(valid_job_requirements)
    pair = ResumeJobPair(
        resume=valid_resume,
        job_description=job,
        fit_level="good",
        trace_id="pair-test-002",
    )

    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    (input_dir / "jobs_20260101_000001.jsonl").write_text(
        job.model_dump_json() + "\n", encoding="utf-8"
    )
    (input_dir / "resumes_20260101_000001.jsonl").write_text(
        valid_resume.model_dump_json() + "\n", encoding="utf-8"
    )
    (input_dir / "pairs_20260101_000001.jsonl").write_text(
        pair.model_dump_json() + "\n", encoding="utf-8"
    )

    result = validator.run_validation(input_dir=input_dir)
    output_paths = validator.write_outputs(result=result, output_dir=tmp_path / "out")

    assert output_paths["validated"].exists()
    assert output_paths["invalid"].exists()
    assert output_paths["failure_modes"].exists()

    validated_payload = json.loads(output_paths["validated"].read_text(encoding="utf-8"))
    failure_modes_payload = json.loads(output_paths["failure_modes"].read_text(encoding="utf-8"))

    assert validated_payload["summary"]["invalid_records"] == 0
    assert set(validated_payload["records"].keys()) == {"jobs", "resumes", "pairs"}
    assert "error_type_distribution" in failure_modes_payload
    assert set(failure_modes_payload["error_type_distribution"].keys()) == set(
        validator.ERROR_CATEGORIES
    )
