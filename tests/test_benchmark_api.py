"""Tests for benchmark_api.py latency metric helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import benchmark_api


def test_build_latency_summary_marks_target_pass() -> None:
    summary = benchmark_api.build_latency_summary([0.5, 0.8, 0.9, 1.1], target_s=2.0)

    assert summary["sample_size"] == 4
    assert summary["passes_target"] is True
    assert summary["p95_s"] <= 2.0


def test_build_latency_summary_marks_target_fail() -> None:
    summary = benchmark_api.build_latency_summary([9.5, 10.5, 11.0], target_s=10.0)

    assert summary["sample_size"] == 3
    assert summary["passes_target"] is False
    assert summary["p95_s"] > 10.0


def test_benchmark_mode_uses_posted_latencies(monkeypatch: pytest.MonkeyPatch) -> None:
    samples = iter([(0.4, 0.35), (0.6, 0.5), (0.7, 0.65)])

    def _stub_post_review_resume(**_kwargs):
        return next(samples)

    monkeypatch.setattr(benchmark_api, "post_review_resume", _stub_post_review_resume)

    payload = {"trace_id": "bench-1"}
    result = benchmark_api.benchmark_mode(
        base_url="http://127.0.0.1:8000",
        payload=payload,
        sample_size=3,
        use_judge=False,
        judge_model_name="gpt-4.1-mini",
        timeout_seconds=5.0,
        target_s=2.0,
    )

    assert result["client"]["sample_size"] == 3
    assert result["api_reported"]["sample_size"] == 3
    assert result["client"]["passes_target"] is True
    assert result["api_reported"]["passes_target"] is True


def test_main_writes_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload_path = tmp_path / "payload.json"
    payload_path.write_text(json.dumps({"trace_id": "bench-main"}), encoding="utf-8")

    def _stub_benchmark_mode(**_kwargs):
        return {
            "client": {
                "sample_size": 2,
                "target_s": 2.0,
                "passes_target": True,
                "mean_s": 0.6,
                "p50_s": 0.6,
                "p95_s": 0.7,
                "max_s": 0.8,
            },
            "api_reported": {
                "sample_size": 2,
                "target_s": 2.0,
                "passes_target": True,
                "mean_s": 0.5,
                "p50_s": 0.5,
                "p95_s": 0.6,
                "max_s": 0.6,
            },
        }

    monkeypatch.setattr(benchmark_api, "benchmark_mode", _stub_benchmark_mode)
    monkeypatch.setattr(
        "sys.argv",
        [
            "benchmark_api.py",
            "--payload-json",
            str(payload_path),
            "--requests-per-mode",
            "2",
            "--output-dir",
            str(tmp_path),
        ],
    )

    benchmark_api.main()

    artifacts = sorted(tmp_path.glob("api_latency_*.json"))
    assert artifacts

    payload = json.loads(artifacts[-1].read_text(encoding="utf-8"))
    assert payload["endpoint"] == "/review-resume"
    assert payload["rules_only"]["client"]["passes_target"] is True
    assert payload["with_judge"]["client"]["passes_target"] is True
