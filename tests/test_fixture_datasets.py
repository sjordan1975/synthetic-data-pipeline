"""Sanity tests for curated fixture datasets used in analyzer evaluation."""

from __future__ import annotations

import json
from pathlib import Path

from schemas.models import ResumeJobPair


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        rows.append(json.loads(raw))
    return rows


def _assert_fixture_schema(path: Path, expected_count: int) -> None:
    rows = _read_jsonl(path)
    assert len(rows) == expected_count

    trace_ids: set[str] = set()
    for row in rows:
        pair = ResumeJobPair.model_validate(row)
        trace_ids.add(pair.trace_id)

    assert len(trace_ids) == expected_count


def test_adversarial_fixture_is_schema_valid() -> None:
    fixture_path = Path(__file__).parent / "fixtures" / "adversarial_pairs_v1.jsonl"
    _assert_fixture_schema(fixture_path, expected_count=7)


def test_control_fixture_is_schema_valid() -> None:
    fixture_path = Path(__file__).parent / "fixtures" / "control_pairs_v1.jsonl"
    _assert_fixture_schema(fixture_path, expected_count=4)
