"""Shared retry and pacing helpers for OpenAI-compatible LLM calls.

This module centralizes explicit rate-limit handling across generation,
correction, and judge paths.
"""

from __future__ import annotations

import os
import random
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


DEFAULT_RETRY_MAX_ATTEMPTS = int(os.getenv("LLM_RETRY_MAX_ATTEMPTS", "5"))
DEFAULT_RETRY_BASE_DELAY_SECONDS = float(os.getenv("LLM_RETRY_BASE_DELAY_SECONDS", "0.5"))
DEFAULT_RETRY_MAX_DELAY_SECONDS = float(os.getenv("LLM_RETRY_MAX_DELAY_SECONDS", "8.0"))
DEFAULT_RETRY_JITTER_RATIO = float(os.getenv("LLM_RETRY_JITTER_RATIO", "0.2"))
DEFAULT_BATCH_DELAY_SECONDS = float(os.getenv("LLM_BATCH_DELAY_SECONDS", "0.0"))


def _retry_after_seconds(exc: Exception) -> float | None:
    """Extract Retry-After seconds from provider exceptions when present."""
    response = getattr(exc, "response", None)
    if response is None:
        return None

    headers = getattr(response, "headers", None)
    if not headers:
        return None

    retry_after = headers.get("retry-after") or headers.get("Retry-After")
    if retry_after is None:
        return None

    try:
        return max(0.0, float(retry_after))
    except (TypeError, ValueError):
        return None


def _status_code(exc: Exception) -> int | None:
    """Best-effort status code extraction across SDK/client exception shapes."""
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code

    response = getattr(exc, "response", None)
    if response is not None:
        response_status = getattr(response, "status_code", None)
        if isinstance(response_status, int):
            return response_status

    return None


def is_retriable_exception(exc: Exception) -> bool:
    """Return True for transient/network/rate-limit exceptions worth retrying."""
    code = _status_code(exc)
    if code == 429:
        return True
    if code is not None and 500 <= code <= 599:
        return True

    name = exc.__class__.__name__.lower()
    if "ratelimit" in name or "timeout" in name or "connection" in name:
        return True

    message = str(exc).lower()
    if "rate limit" in message or "too many requests" in message:
        return True

    return False


def call_with_backoff(
    operation: Callable[[], T],
    *,
    max_attempts: int = DEFAULT_RETRY_MAX_ATTEMPTS,
    base_delay_seconds: float = DEFAULT_RETRY_BASE_DELAY_SECONDS,
    max_delay_seconds: float = DEFAULT_RETRY_MAX_DELAY_SECONDS,
    jitter_ratio: float = DEFAULT_RETRY_JITTER_RATIO,
    sleep_fn: Callable[[float], None] = time.sleep,
    random_fn: Callable[[], float] = random.random,
) -> T:
    """Execute operation with exponential backoff for retriable failures."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except Exception as exc:
            should_retry = is_retriable_exception(exc) and attempt < max_attempts
            if not should_retry:
                raise

            retry_after = _retry_after_seconds(exc)
            if retry_after is None:
                delay = min(max_delay_seconds, base_delay_seconds * (2 ** (attempt - 1)))
            else:
                delay = min(max_delay_seconds, retry_after)

            jitter = delay * jitter_ratio * ((2.0 * random_fn()) - 1.0)
            sleep_fn(max(0.0, delay + jitter))

    raise RuntimeError("Unreachable retry state")


def maybe_batch_delay(
    delay_seconds: float | None = None,
    *,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> None:
    """Apply optional fixed delay between batch requests to smooth request bursts."""
    actual_delay = DEFAULT_BATCH_DELAY_SECONDS if delay_seconds is None else delay_seconds
    if actual_delay > 0:
        sleep_fn(actual_delay)
