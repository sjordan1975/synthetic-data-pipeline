"""Tests for Phase 7 FastAPI service endpoints."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

import api
from schemas.models import JobDescription, ResumeJobPair


class StubJudgeClient:
    """Deterministic judge stub for endpoint tests."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def evaluate_pair(self, pair: ResumeJobPair, rule_analysis: object) -> dict:
        return {
            "has_hallucinations": False,
            "hallucination_details": "",
            "has_awkward_language": False,
            "awkward_language_details": "",
            "overall_quality_score": 0.88,
            "fit_assessment": "good fit",
            "recommendations": ["Keep emphasizing measurable impact."],
            "red_flags": [],
        }


def _build_pair_payload(valid_resume, valid_job_requirements, trace_id: str = "api-test-001") -> dict:
    job = JobDescription(
        company="API Startup",
        industry="Software",
        company_size="51-200",
        position="Software Engineer",
        location="San Francisco, CA",
        description="Build backend APIs and support production systems.",
        requirements=valid_job_requirements,
        metadata={
            "trace_id": "job-api-test-001",
            "generated_at": "2026-03-01T00:00:00Z",
            "is_niche_role": False,
            "used_fallback": False,
        },
    )

    pair = ResumeJobPair(
        resume=valid_resume,
        job_description=job,
        fit_level="good",
        trace_id=trace_id,
    )
    return pair.model_dump(mode="json")


def test_health_endpoint_returns_ok() -> None:
    client = TestClient(api.app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_review_resume_without_judge_returns_analysis(valid_resume, valid_job_requirements) -> None:
    client = TestClient(api.app)
    payload = _build_pair_payload(valid_resume, valid_job_requirements)

    response = client.post("/review-resume", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["trace_id"] == "api-test-001"
    assert body["analysis"]["computed_fit_label"] in {"excellent", "good", "partial", "poor", "complete_mismatch"}
    assert body["adjudication"]["policy_id"] == "benchmark_policy_v1"
    assert body["judge"] is None


def test_review_resume_with_judge_returns_judge_payload(valid_resume, valid_job_requirements, monkeypatch) -> None:
    monkeypatch.setattr(api, "OpenAIJudgeClient", StubJudgeClient)
    client = TestClient(api.app)
    payload = _build_pair_payload(valid_resume, valid_job_requirements, trace_id="api-test-judge")

    response = client.post("/review-resume?use_judge=true", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["trace_id"] == "api-test-judge"
    assert body["judge"] is not None
    assert body["judge"]["overall_quality_score"] == 0.88


def test_review_resume_bad_input_returns_400() -> None:
    client = TestClient(api.app)

    response = client.post("/review-resume", json={})

    assert response.status_code == 400
    body = response.json()
    assert body["error"] == "invalid_request"


def test_failure_rates_endpoint_returns_aggregates(tmp_path: Path) -> None:
    artifact = tmp_path / "failure_labels_test.jsonl"
    rows = [
        {
            "trace_id": "a",
            "computed_fit_label": "good",
            "experience_mismatch": False,
            "seniority_mismatch": False,
            "missing_core_skills": True,
            "hallucinated_skills": False,
            "awkward_language": True,
            "requires_human_review": True,
        },
        {
            "trace_id": "b",
            "computed_fit_label": "partial",
            "experience_mismatch": True,
            "seniority_mismatch": False,
            "missing_core_skills": False,
            "hallucinated_skills": False,
            "awkward_language": False,
            "requires_human_review": False,
        },
    ]
    artifact.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    client = TestClient(api.app)
    response = client.get(f"/analysis/failure-rates?source_file={artifact}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["row_count"] == 2
    assert payload["failure_rates"]["experience_mismatch"] == 0.5
    assert payload["failure_rates"]["missing_core_skills"] == 0.5
    assert payload["fit_label_distribution"] == {"good": 1, "partial": 1}


def test_failure_rates_returns_404_when_no_artifact(monkeypatch) -> None:
    monkeypatch.setattr(api, "find_latest_file", lambda *_args, **_kwargs: None)
    client = TestClient(api.app)

    response = client.get("/analysis/failure-rates")

    assert response.status_code == 404
    assert response.json()["error"] == "request_error"
