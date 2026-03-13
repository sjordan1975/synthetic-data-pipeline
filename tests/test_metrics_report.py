"""Tests for metrics_report.py markdown generator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import metrics_report


def test_metrics_report_writes_markdown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    metrics_dir = tmp_path / "outputs" / "metrics"
    outputs_dir = tmp_path / "outputs"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    api_latency = {
        "rules_only": {"client": {"p95_s": 0.9, "target_s": 2.0}},
        "with_judge": {"client": {"p95_s": 8.2, "target_s": 10.0}},
    }
    (metrics_dir / "api_latency_20260101_010101.json").write_text(
        json.dumps(api_latency), encoding="utf-8"
    )

    pipeline_summary = {"summary": {"correction_success_rate": 0.65}}
    (outputs_dir / "pipeline_summary_20260101_010102.json").write_text(
        json.dumps(pipeline_summary), encoding="utf-8"
    )

    stage_times = {
        "durations_seconds": {
            "generation": 1.23,
            "validation": 0.45,
            "analysis": 0.67,
        }
    }
    stage_times_path = outputs_dir / "stage_times.json"
    stage_times_path.write_text(json.dumps(stage_times), encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "metrics_report.py",
            "--metrics-dir",
            str(metrics_dir),
            "--pipeline-summary-dir",
            str(outputs_dir),
            "--stage-times-json",
            str(stage_times_path),
            "--correction-target",
            "0.5",
        ],
    )

    metrics_report.main()

    reports = sorted(metrics_dir.glob("metrics_report_*.md"))
    assert reports

    text = reports[-1].read_text(encoding="utf-8")
    assert "Rules-only p95 latency" in text
    assert "With-judge p95 latency" in text
    assert "Correction success rate" in text
    assert "PASS" in text


def test_metrics_report_marks_correction_not_run_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metrics_dir = tmp_path / "outputs" / "metrics"
    outputs_dir = tmp_path / "outputs"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    api_latency = {
        "rules_only": {"client": {"p95_s": 0.9, "target_s": 2.0}},
        "with_judge": {"client": {"p95_s": 8.2, "target_s": 10.0}},
    }
    (metrics_dir / "api_latency_20260101_020201.json").write_text(
        json.dumps(api_latency), encoding="utf-8"
    )

    # Pipeline summary intentionally omits correction_success_rate.
    pipeline_summary = {"summary": {"validation_success_rate": 0.98}}
    (outputs_dir / "pipeline_summary_20260101_020202.json").write_text(
        json.dumps(pipeline_summary), encoding="utf-8"
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "metrics_report.py",
            "--metrics-dir",
            str(metrics_dir),
            "--pipeline-summary-dir",
            str(outputs_dir),
            "--stage-times-json",
            str(outputs_dir / "missing_stage_times.json"),
        ],
    )

    metrics_report.main()

    reports = sorted(metrics_dir.glob("metrics_report_*.md"))
    assert reports

    text = reports[-1].read_text(encoding="utf-8")
    assert "Correction success rate: n/a vs target 50.0% -> NOT RUN" in text
    assert "Correction loop evidence: not run or not available in pipeline summary." in text


def test_metrics_report_marks_api_mode_not_run_when_missing_p95(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metrics_dir = tmp_path / "outputs" / "metrics"
    outputs_dir = tmp_path / "outputs"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Simulate partial benchmark output where with_judge data is missing.
    api_latency = {
        "rules_only": {"client": {"p95_s": 1.1, "target_s": 2.0}},
        "with_judge": {"client": {"target_s": 10.0}},
    }
    (metrics_dir / "api_latency_20260101_030301.json").write_text(
        json.dumps(api_latency), encoding="utf-8"
    )

    pipeline_summary = {"summary": {"correction_success_rate": 0.61}}
    (outputs_dir / "pipeline_summary_20260101_030302.json").write_text(
        json.dumps(pipeline_summary), encoding="utf-8"
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "metrics_report.py",
            "--metrics-dir",
            str(metrics_dir),
            "--pipeline-summary-dir",
            str(outputs_dir),
            "--stage-times-json",
            str(outputs_dir / "missing_stage_times.json"),
        ],
    )

    metrics_report.main()

    reports = sorted(metrics_dir.glob("metrics_report_*.md"))
    assert reports

    text = reports[-1].read_text(encoding="utf-8")
    assert "Rules-only p95 latency: 1.100s vs target 2.000s -> PASS" in text
    assert "With-judge p95 latency: n/a vs target 10.000s -> NOT RUN" in text
