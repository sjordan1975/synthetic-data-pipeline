"""Tests for capture_stage_time.py stage runtime artifact writer."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import capture_stage_time


class _CompletedProcess:
    """Minimal completed process stub."""

    def __init__(self, returncode: int) -> None:
        self.returncode = returncode


def test_capture_stage_time_writes_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    times_path = tmp_path / "stage_times.json"

    def _stub_run(_command, check):
        assert check is False
        return _CompletedProcess(returncode=0)

    monkeypatch.setattr(subprocess, "run", _stub_run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "capture_stage_time.py",
            "--stage",
            "generation",
            "--times-json",
            str(times_path),
            "--",
            "python",
            "generator.py",
        ],
    )

    capture_stage_time.main()

    payload = json.loads(times_path.read_text(encoding="utf-8"))
    assert "durations_seconds" in payload
    assert payload["durations_seconds"]["generation"] >= 0.0
    assert payload["stage_metadata"]["generation"]["exit_code"] == 0


def test_capture_stage_time_rejects_invalid_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "capture_stage_time.py",
            "--stage",
            "invalid",
            "--",
            "python",
            "validator.py",
        ],
    )

    with pytest.raises(ValueError):
        capture_stage_time.main()
