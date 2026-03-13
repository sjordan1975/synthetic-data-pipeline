"""Phase 7 FastAPI service for real-time resume-job analysis."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

from analyzer import (
    DEFAULT_JUDGE_MODEL_NAME,
    OpenAIJudgeClient,
    analyze_pairs,
    find_latest_file,
)
from schemas.models import ResumeJobPair

app = FastAPI(
    title="Resume Coach API",
    description="Real-time resume/job analysis with optional LLM-as-Judge enrichment.",
    version="1.0.0",
)


class ReviewResumeResponse(BaseModel):
    """Response schema for POST /review-resume."""

    trace_id: str
    analysis: dict[str, Any]
    adjudication: dict[str, Any]
    judge: dict[str, Any] | None = None
    latency_ms: float


class FailureRatesResponse(BaseModel):
    """Response schema for GET /analysis/failure-rates."""

    source_file: str
    row_count: int
    failure_rates: dict[str, float]
    fit_label_distribution: dict[str, int]


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    _request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Map request validation errors to rubric-required HTTP 400 payloads."""
    return JSONResponse(
        status_code=400,
        content={
            "error": "invalid_request",
            "message": "Request body or query parameters are invalid.",
            "details": exc.errors(),
        },
    )


@app.exception_handler(ValidationError)
async def pydantic_validation_exception_handler(
    _request: Request,
    exc: ValidationError,
) -> JSONResponse:
    """Map explicit model validation failures to HTTP 400."""
    return JSONResponse(
        status_code=400,
        content={
            "error": "invalid_request",
            "message": "Payload validation failed.",
            "details": exc.errors(),
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(_request: Request, exc: HTTPException) -> JSONResponse:
    """Ensure HTTPException responses are always JSON objects."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "request_error", "message": str(exc.detail)},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Guardrail for unexpected failures."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "server_error",
            "message": "Unexpected server error.",
            "details": str(exc),
        },
    )


def _load_failure_label_rows(path: Path) -> list[dict[str, Any]]:
    """Read failure_labels JSONL records from disk."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            rows.append(json.loads(raw))
    return rows


def _build_failure_rates(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Compute aggregate failure rates across key analyzer signals."""
    keys = [
        "experience_mismatch",
        "seniority_mismatch",
        "missing_core_skills",
        "hallucinated_skills",
        "awkward_language",
        "requires_human_review",
    ]
    total = len(rows)
    if total == 0:
        return {key: 0.0 for key in keys}

    rates: dict[str, float] = {}
    for key in keys:
        count = sum(1 for row in rows if bool(row.get(key)))
        rates[key] = round(count / total, 4)
    return rates


def _fit_label_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Count computed fit labels in failure label rows."""
    distribution: dict[str, int] = {}
    for row in rows:
        label = str(row.get("computed_fit_label", "unknown"))
        distribution[label] = distribution.get(label, 0) + 1
    return distribution


@app.get("/health")
def health() -> dict[str, str]:
    """Lightweight service health endpoint."""
    return {"status": "ok", "service": "resume-coach-api"}


@app.post("/review-resume", response_model=ReviewResumeResponse)
def review_resume(
    pair: ResumeJobPair,
    use_judge: bool = Query(default=False, description="Enable optional LLM-as-Judge."),
    judge_model_name: str = Query(
        default=DEFAULT_JUDGE_MODEL_NAME,
        description="Model name used when use_judge=true.",
    ),
) -> ReviewResumeResponse:
    """Analyze one resume/job pair with optional LLM judge enrichment."""
    start = time.perf_counter()

    judge_client: OpenAIJudgeClient | None = None
    if use_judge:
        try:
            judge_client = OpenAIJudgeClient(model_name=judge_model_name)
        except ValueError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    labels, adjudications, judge_rows = analyze_pairs([pair], judge_client=judge_client)
    label = labels[0]
    adjudication = adjudications[0]
    judge = judge_rows[0] if judge_rows else None

    latency_ms = round((time.perf_counter() - start) * 1000.0, 2)
    return ReviewResumeResponse(
        trace_id=pair.trace_id,
        analysis=label,
        adjudication=adjudication,
        judge=judge,
        latency_ms=latency_ms,
    )


@app.get("/analysis/failure-rates", response_model=FailureRatesResponse)
def failure_rates(
    source_file: str | None = Query(
        default=None,
        description="Optional explicit failure_labels JSONL path.",
    ),
) -> FailureRatesResponse:
    """Return aggregate failure rates from the latest (or explicit) analysis artifact."""
    target_path: Path
    if source_file:
        target_path = Path(source_file)
    else:
        latest = find_latest_file(Path("outputs/analysis"), "failure_labels_*.jsonl")
        if latest is None:
            raise HTTPException(status_code=404, detail="No failure_labels artifacts found in outputs/analysis.")
        target_path = latest

    if not target_path.exists():
        raise HTTPException(status_code=404, detail=f"Failure labels artifact not found: {target_path}")

    rows = _load_failure_label_rows(target_path)
    if not rows:
        raise HTTPException(status_code=404, detail=f"No rows found in artifact: {target_path}")

    return FailureRatesResponse(
        source_file=str(target_path),
        row_count=len(rows),
        failure_rates=_build_failure_rates(rows),
        fit_label_distribution=_fit_label_distribution(rows),
    )
