"""Targeted tests for Phase 7 correction loop behavior."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import corrector


class StubCorrectionClient:
    """Deterministic fake correction client for retry-loop tests."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self.calls: list[str] = []

    def generate_correction(self, prompt: str) -> str:
        self.calls.append(prompt)
        if self._responses:
            return self._responses.pop(0)
        return "{}"


@pytest.fixture
def invalid_resume_record() -> dict:
    return {
        "source_file": "data/generated/resumes_20260101_000000.jsonl",
        "record_type": "resumes",
        "line_number": 1,
        "error_count": 1,
        "error_categories": ["missing_required_fields"],
        "errors": [
            {
                "loc": "contact_info",
                "type": "missing",
                "msg": "Field required",
                "category": "missing_required_fields",
            }
        ],
        "raw_record": {
            "education": [
                {
                    "institution": "Uni",
                    "degree": "Bachelor of Science",
                    "field_of_study": "Computer Science",
                    "graduation_year": 2020,
                    "gpa": 3.5,
                }
            ],
            "experience": [
                {
                    "company": "Tech Corp",
                    "position": "Engineer",
                    "start_date": "2021-01-01",
                    "end_date": "2023-01-01",
                    "description": "Built APIs and maintained backend services.",
                    "achievements": ["Reduced errors by 20%"],
                }
            ],
            "skills": [
                {"name": "Python", "category": "Programming", "proficiency": "expert"},
                {"name": "SQL", "category": "Programming", "proficiency": "advanced"},
                {"name": "Docker", "category": "DevOps", "proficiency": "intermediate"},
            ],
            "metadata": {
                "trace_id": "resume-test-404",
                "generated_at": "2024-01-01T00:00:00Z",
                "prompt_template": "resume_professional_v1",
                "fit_level": "good",
                "writing_style": "professional",
                "used_fallback": False,
            },
        },
    }


def build_valid_resume_payload() -> dict:
    return {
        "contact_info": {
            "name": "Jordan Kim",
            "email": "jordan.kim@example.com",
            "phone_region": "US",
            "phone": "+12025550123",
            "location": "San Francisco, CA",
        },
        "education": [
            {
                "institution": "Uni",
                "degree": "Bachelor of Science",
                "field_of_study": "Computer Science",
                "graduation_year": 2020,
                "gpa": 3.5,
            }
        ],
        "experience": [
            {
                "company": "Tech Corp",
                "position": "Engineer",
                "start_date": "2021-01-01",
                "end_date": "2023-01-01",
                "description": "Built APIs and maintained backend services.",
                "achievements": ["Reduced errors by 20%"],
            }
        ],
        "skills": [
            {"name": "Python", "category": "Programming", "proficiency": "expert"},
            {"name": "SQL", "category": "Programming", "proficiency": "advanced"},
            {"name": "Docker", "category": "DevOps", "proficiency": "intermediate"},
        ],
        "summary": "Backend engineer with API and data systems experience.",
        "metadata": {
            "trace_id": "resume-test-404",
            "generated_at": "2024-01-01T00:00:00Z",
            "prompt_template": "resume_professional_v1",
            "fit_level": "good",
            "writing_style": "professional",
            "used_fallback": False,
        },
    }


def test_extract_json_object_handles_fenced_output() -> None:
    text = """```json
+{"a": 1}
+```""".replace("+", "")
    payload = corrector.extract_json_object(text)
    assert payload == {"a": 1}


def test_run_correction_loop_succeeds_on_second_try(invalid_resume_record: dict) -> None:
    valid_payload = build_valid_resume_payload()
    client = StubCorrectionClient(
        responses=[
            "not json at all",
            json.dumps(valid_payload),
        ]
    )

    outcomes = corrector.run_correction_loop(
        invalid_records=[invalid_resume_record],
        client=client,
        max_attempts=3,
    )

    assert len(outcomes) == 1
    outcome = outcomes[0]
    assert outcome.succeeded
    assert len(outcome.attempts) == 2
    assert outcome.attempts[0].succeeded is False
    assert outcome.attempts[1].succeeded is True


def test_run_correction_loop_marks_permanent_failure(invalid_resume_record: dict) -> None:
    client = StubCorrectionClient(
        responses=[
            "{}",
            "{}",
            "{}",
        ]
    )

    outcomes = corrector.run_correction_loop(
        invalid_records=[invalid_resume_record],
        client=client,
        max_attempts=3,
    )

    outcome = outcomes[0]
    assert outcome.succeeded is False
    assert len(outcome.attempts) == 3
    assert all(attempt.succeeded is False for attempt in outcome.attempts)


def test_compute_correction_report_tracks_success_and_attempts(
    invalid_resume_record: dict,
    tmp_path: Path,
) -> None:
    ok = corrector.CorrectionResult(
        source_record=invalid_resume_record,
        corrected_payload=build_valid_resume_payload(),
        attempts=[
            corrector.CorrectionAttempt(
                attempt_number=1,
                succeeded=True,
                error_categories=[],
                errors=[],
            )
        ],
    )
    failed = corrector.CorrectionResult(
        source_record=invalid_resume_record,
        corrected_payload=None,
        attempts=[
            corrector.CorrectionAttempt(
                attempt_number=1,
                succeeded=False,
                error_categories=["missing_required_fields"],
                errors=[],
            ),
            corrector.CorrectionAttempt(
                attempt_number=2,
                succeeded=False,
                error_categories=["type_mismatches"],
                errors=[],
            ),
        ],
    )

    report = corrector.compute_correction_report(
        outcomes=[ok, failed],
        max_attempts=3,
        input_source=tmp_path / "invalid_20260101_000000.jsonl",
    )

    assert report["summary"]["total_invalid_records"] == 2
    assert report["summary"]["corrected_records"] == 1
    assert report["summary"]["permanently_failed_records"] == 1
    assert report["summary"]["correction_success_rate"] == 0.5
    assert report["summary"]["average_attempts"] == 1.5


def test_write_correction_outputs_emits_all_artifacts(
    invalid_resume_record: dict,
    tmp_path: Path,
) -> None:
    outcomes = [
        corrector.CorrectionResult(
            source_record=invalid_resume_record,
            corrected_payload=build_valid_resume_payload(),
            attempts=[
                corrector.CorrectionAttempt(
                    attempt_number=1,
                    succeeded=True,
                    error_categories=[],
                    errors=[],
                )
            ],
        ),
        corrector.CorrectionResult(
            source_record=invalid_resume_record,
            corrected_payload=None,
            attempts=[
                corrector.CorrectionAttempt(
                    attempt_number=1,
                    succeeded=False,
                    error_categories=["missing_required_fields"],
                    errors=[{"msg": "Field required"}],
                )
            ],
        ),
    ]

    report = {
        "summary": {
            "total_invalid_records": 2,
            "corrected_records": 1,
            "permanently_failed_records": 1,
            "correction_success_rate": 0.5,
            "average_attempts": 1.0,
            "meets_success_criteria_gt_50pct": False,
        }
    }

    paths = corrector.write_correction_outputs(
        outcomes=outcomes,
        report=report,
        output_dir=tmp_path / "out",
    )

    assert paths["corrected_valid"].exists()
    assert paths["corrected_still_invalid"].exists()
    assert paths["report"].exists()
