"""Unit tests for shared LLM retry/backoff helpers."""

from __future__ import annotations

import pytest

import llm_retry


class _FakeResponse:
    def __init__(self, status_code: int | None = None, headers: dict[str, str] | None = None) -> None:
        self.status_code = status_code
        self.headers = headers or {}


class _FakeError(Exception):
    def __init__(self, message: str, *, status_code: int | None = None, headers: dict[str, str] | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response = _FakeResponse(status_code=status_code, headers=headers)


def test_call_with_backoff_retries_rate_limit_then_succeeds() -> None:
    attempts = {"count": 0}
    sleeps: list[float] = []

    def _operation() -> str:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise _FakeError("Too many requests", status_code=429)
        return "ok"

    result = llm_retry.call_with_backoff(
        _operation,
        max_attempts=4,
        base_delay_seconds=0.01,
        max_delay_seconds=0.1,
        jitter_ratio=0.0,
        sleep_fn=sleeps.append,
    )

    assert result == "ok"
    assert attempts["count"] == 3
    assert sleeps == [0.01, 0.02]


def test_call_with_backoff_uses_retry_after_header_when_present() -> None:
    sleeps: list[float] = []

    def _operation() -> str:
        raise _FakeError("Rate limited", status_code=429, headers={"Retry-After": "1.5"})

    with pytest.raises(_FakeError):
        llm_retry.call_with_backoff(
            _operation,
            max_attempts=2,
            base_delay_seconds=0.01,
            max_delay_seconds=2.0,
            jitter_ratio=0.0,
            sleep_fn=sleeps.append,
        )

    assert sleeps == [1.5]


def test_call_with_backoff_does_not_retry_non_retriable_errors() -> None:
    attempts = {"count": 0}

    def _operation() -> str:
        attempts["count"] += 1
        raise ValueError("schema mismatch")

    with pytest.raises(ValueError):
        llm_retry.call_with_backoff(_operation, max_attempts=5, sleep_fn=lambda _x: None)

    assert attempts["count"] == 1


def test_maybe_batch_delay_applies_only_positive_values() -> None:
    sleeps: list[float] = []

    llm_retry.maybe_batch_delay(delay_seconds=0.0, sleep_fn=sleeps.append)
    llm_retry.maybe_batch_delay(delay_seconds=0.05, sleep_fn=sleeps.append)

    assert sleeps == [0.05]
