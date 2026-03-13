"""Sync tests ensuring analyzer constants stay aligned with benchmark policy YAML.

Policy file is documentation/spec in this project version (not runtime-loaded),
so these tests guard against drift.
"""

from __future__ import annotations

import re
from pathlib import Path

import analyzer
from schemas.models import FitLevel


POLICY_FILE = Path("config/benchmark_policy_v1.yaml")


def read_policy_text() -> str:
    text = POLICY_FILE.read_text(encoding="utf-8")
    assert text.strip(), "benchmark policy file is unexpectedly empty"
    return text


def extract_float_pair(text: str, section: str) -> tuple[float, float]:
    pattern = (
        rf"{section}:\n"
        rf"\s+min_overlap:\s*([0-9.]+)\n"
        rf"\s+max_overlap:\s*([0-9.]+)"
    )
    match = re.search(pattern, text)
    assert match is not None, f"Could not find overlap bounds for section: {section}"
    return float(match.group(1)), float(match.group(2))


def extract_int(text: str, key: str) -> int:
    match = re.search(rf"{key}:\s*([0-9]+)", text)
    assert match is not None, f"Could not find integer value for key: {key}"
    return int(match.group(1))


def extract_float(text: str, key: str) -> float:
    match = re.search(rf"{key}:\s*([0-9]+(?:\.[0-9]+)?)", text)
    assert match is not None, f"Could not find float value for key: {key}"
    return float(match.group(1))


def extract_phrase_list(text: str, section: str) -> list[str]:
    block_match = re.search(rf"{section}:\n((?:\s+-\s+\".*\"\n)+)", text)
    assert block_match is not None, f"Could not find list block for section: {section}"
    lines = block_match.group(1).strip().splitlines()
    return [line.split("-", 1)[1].strip().strip('"') for line in lines]


def test_fit_band_constants_match_policy_file() -> None:
    text = read_policy_text()

    expected = {
        FitLevel.EXCELLENT: extract_float_pair(text, "excellent"),
        FitLevel.GOOD: extract_float_pair(text, "good"),
        FitLevel.PARTIAL: extract_float_pair(text, "partial"),
        FitLevel.POOR: extract_float_pair(text, "poor"),
        FitLevel.COMPLETE_MISMATCH: extract_float_pair(text, "complete_mismatch"),
    }

    assert analyzer.FIT_BANDS == expected


def test_hallucination_threshold_constants_match_policy_file() -> None:
    text = read_policy_text()

    assert analyzer.EXPERT_SKILL_MAX_FOR_ENTRY_LEVEL == extract_int(
        text, "expert_skill_max_for_entry_level"
    )
    assert analyzer.TOTAL_SKILL_MAX_FOR_ENTRY_LEVEL == extract_int(
        text, "total_skill_max_for_entry_level"
    )
    assert list(analyzer.SUSPICIOUS_PHRASES) == extract_phrase_list(text, "suspicious_phrases")


def test_awkward_language_constants_match_policy_file() -> None:
    text = read_policy_text()

    assert analyzer.BUZZWORD_THRESHOLD == extract_int(text, "buzzword_threshold")
    assert analyzer.REPEATED_TOKEN_WINDOW == extract_int(text, "repeated_token_window")
    assert analyzer.REPEATED_TOKEN_COUNT == extract_int(text, "repeated_token_count")


def test_adjudication_constants_match_policy_file() -> None:
    text = read_policy_text()

    assert analyzer.POLICY_ID in text
    assert analyzer.OVERLAP_MARGIN_FROM_BOUNDARY_MAX == extract_float(
        text, "overlap_margin_from_boundary_max"
    )
    assert analyzer.RANDOM_AUDIT_RATE == extract_float(text, "random_audit_rate")

    for field in analyzer.ADJUDICATION_OUTPUT_FIELDS:
        assert re.search(rf"-\s+{re.escape(field)}\b", text), (
            f"Missing adjudication output field '{field}' in policy file"
        )
