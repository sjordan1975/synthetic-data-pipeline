"""Tests for summarize_pipeline.py artifact aggregation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import summarize_pipeline


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_summarize_pipeline_writes_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    generated_dir = tmp_path / "data" / "generated"
    validation_dir = tmp_path / "outputs" / "validation"
    analysis_dir = tmp_path / "outputs" / "analysis"
    corrections_dir = tmp_path / "outputs" / "corrections"
    metrics_dir = tmp_path / "outputs" / "metrics"
    summary_dir = tmp_path / "outputs"

    for directory in (generated_dir, validation_dir, analysis_dir, corrections_dir, metrics_dir):
        directory.mkdir(parents=True, exist_ok=True)

    _write_jsonl(generated_dir / "jobs_20260101_000001.jsonl", [{"id": "job-1"}, {"id": "job-2"}])
    _write_jsonl(
        generated_dir / "resumes_20260101_000001.jsonl",
        [{"id": "resume-1"}, {"id": "resume-2"}, {"id": "resume-3"}],
    )
    _write_jsonl(
        generated_dir / "pairs_20260101_000001.jsonl",
        [{"id": "pair-1"}, {"id": "pair-2"}, {"id": "pair-3"}],
    )

    validated_data = {
        "summary": {
            "total_records": 8,
            "valid_records": 8,
            "invalid_records": 0,
            "validation_success_rate": 1.0,
        }
    }
    (validation_dir / "validated_data_20260101_000002.json").write_text(
        json.dumps(validated_data), encoding="utf-8"
    )

    failure_rows = [
        {
            "experience_mismatch": True,
            "seniority_mismatch": False,
            "missing_core_skills": True,
            "hallucinated_skills": False,
            "awkward_language": False,
        },
        {
            "experience_mismatch": False,
            "seniority_mismatch": True,
            "missing_core_skills": False,
            "hallucinated_skills": False,
            "awkward_language": True,
        },
    ]
    _write_jsonl(analysis_dir / "failure_labels_20260101_000003.jsonl", failure_rows)

    correction_report = {"summary": {"correction_success_rate": 0.75}}
    (corrections_dir / "correction_report_20260101_000004.json").write_text(
        json.dumps(correction_report), encoding="utf-8"
    )

    api_latency = {
        "endpoint": "/review-resume",
        "sample_size": 4,
        "rules_only": {"client": {"p95_s": 0.5, "target_s": 2.0, "passes_target": True}},
        "with_judge": {"client": {"p95_s": 7.8, "target_s": 10.0, "passes_target": True}},
    }
    (metrics_dir / "api_latency_20260101_000005.json").write_text(
        json.dumps(api_latency), encoding="utf-8"
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "summarize_pipeline.py",
            "--generated-dir",
            str(generated_dir),
            "--validation-dir",
            str(validation_dir),
            "--analysis-dir",
            str(analysis_dir),
            "--corrections-dir",
            str(corrections_dir),
            "--metrics-dir",
            str(metrics_dir),
            "--output-dir",
            str(summary_dir),
        ],
    )

    summarize_pipeline.main()

    summaries = sorted(summary_dir.glob("pipeline_summary_*.json"))
    assert summaries

    payload = json.loads(summaries[-1].read_text(encoding="utf-8"))
    summary = payload["summary"]

    assert summary["total_records_generated"]["jobs"] == 2
    assert summary["total_records_generated"]["resumes"] == 3
    assert summary["total_records_generated"]["pairs"] == 3
    assert summary["validation_success_rate"] == 1.0
    assert summary["failure_mode_distribution"]["row_count"] == 2
    assert summary["failure_mode_distribution"]["flag_counts"]["experience_mismatch"] == 1
    assert summary["correction_success_rate"] == 0.75
    assert summary["api_latency"] is not None
    assert summary["api_latency"]["endpoint"] == "/review-resume"
    assert summary["api_latency"]["with_judge"]["client"]["passes_target"] is True


def test_summarize_pipeline_raises_when_required_artifacts_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generated_dir = tmp_path / "data" / "generated"
    validation_dir = tmp_path / "outputs" / "validation"
    analysis_dir = tmp_path / "outputs" / "analysis"
    summary_dir = tmp_path / "outputs"

    for directory in (generated_dir, validation_dir, analysis_dir):
        directory.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        "sys.argv",
        [
            "summarize_pipeline.py",
            "--generated-dir",
            str(generated_dir),
            "--validation-dir",
            str(validation_dir),
            "--analysis-dir",
            str(analysis_dir),
            "--output-dir",
            str(summary_dir),
        ],
    )

    with pytest.raises(FileNotFoundError):
        summarize_pipeline.main()
