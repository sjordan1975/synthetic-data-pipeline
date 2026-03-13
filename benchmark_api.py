"""Benchmark API latency for /review-resume with and without LLM-as-Judge.

This script measures end-to-end client-side latency and captures API-reported
latency_ms for both modes:
- rules-only (use_judge=false)
- with-judge (use_judge=true)

It writes a timestamped JSON artifact to outputs/metrics/ for use by
summarize_pipeline.py.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def percentile(values: list[float], pct: float) -> float:
    """Compute percentile via linear interpolation.

    Args:
        values: Numeric samples.
        pct: Percentile as [0, 100].

    Returns:
        Percentile value or 0.0 when values are empty.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    rank = (len(ordered) - 1) * (pct / 100.0)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def build_latency_summary(latencies_s: list[float], target_s: float) -> dict[str, Any]:
    """Build aggregate latency stats for one benchmark mode."""
    if not latencies_s:
        return {
            "sample_size": 0,
            "target_s": target_s,
            "passes_target": False,
            "mean_s": 0.0,
            "p50_s": 0.0,
            "p95_s": 0.0,
            "max_s": 0.0,
        }

    p95 = percentile(latencies_s, 95.0)
    return {
        "sample_size": len(latencies_s),
        "target_s": target_s,
        "passes_target": p95 <= target_s,
        "mean_s": round(statistics.mean(latencies_s), 4),
        "p50_s": round(percentile(latencies_s, 50.0), 4),
        "p95_s": round(p95, 4),
        "max_s": round(max(latencies_s), 4),
    }


def post_review_resume(
    *,
    base_url: str,
    payload: dict[str, Any],
    use_judge: bool,
    judge_model_name: str,
    timeout_seconds: float,
) -> tuple[float, float | None]:
    """Send one POST /review-resume request and return client/API latencies.

    Returns:
        Tuple of (client_latency_seconds, api_latency_seconds_or_none).
    """
    query: dict[str, str] = {"use_judge": "true" if use_judge else "false"}
    if use_judge:
        query["judge_model_name"] = judge_model_name

    endpoint = f"{base_url.rstrip('/')}/review-resume?{urllib.parse.urlencode(query)}"
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        method="POST",
        data=body,
        headers={"Content-Type": "application/json"},
    )

    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"HTTP {exc.code} from /review-resume (use_judge={use_judge}): {error_body}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Unable to connect to API at {base_url}. Is uvicorn running?"
        ) from exc

    client_latency_s = time.perf_counter() - start
    payload_json = json.loads(raw)
    latency_ms = payload_json.get("latency_ms")
    api_latency_s = float(latency_ms) / 1000.0 if isinstance(latency_ms, (int, float)) else None
    return client_latency_s, api_latency_s


def benchmark_mode(
    *,
    base_url: str,
    payload: dict[str, Any],
    sample_size: int,
    use_judge: bool,
    judge_model_name: str,
    timeout_seconds: float,
    target_s: float,
) -> dict[str, Any]:
    """Run one latency benchmark mode and return metrics."""
    client_latencies: list[float] = []
    api_latencies: list[float] = []

    for _ in range(sample_size):
        client_latency_s, api_latency_s = post_review_resume(
            base_url=base_url,
            payload=payload,
            use_judge=use_judge,
            judge_model_name=judge_model_name,
            timeout_seconds=timeout_seconds,
        )
        client_latencies.append(client_latency_s)
        if api_latency_s is not None:
            api_latencies.append(api_latency_s)

    return {
        "client": build_latency_summary(client_latencies, target_s),
        "api_reported": build_latency_summary(api_latencies, target_s),
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Benchmark /review-resume latency")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Base URL for the running API service.",
    )
    parser.add_argument(
        "--payload-json",
        type=Path,
        default=Path("tests/fixtures/control_pairs_v1_first_record.json"),
        help="JSON payload file for /review-resume requests.",
    )
    parser.add_argument(
        "--requests-per-mode",
        type=int,
        default=10,
        help="Number of requests per mode (rules-only and with-judge).",
    )
    parser.add_argument(
        "--judge-model-name",
        default="gpt-4.1-mini",
        help="Judge model name used when benchmarking with use_judge=true.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=30.0,
        help="Per-request network timeout.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/metrics"),
        help="Directory where api_latency_<timestamp>.json is written.",
    )
    return parser.parse_args()


def main() -> None:
    """Run API latency benchmark and write JSON artifact."""
    args = parse_args()
    if args.requests_per_mode <= 0:
        raise ValueError("--requests-per-mode must be > 0")

    payload = json.loads(args.payload_json.read_text(encoding="utf-8"))

    rules_only = benchmark_mode(
        base_url=args.base_url,
        payload=payload,
        sample_size=args.requests_per_mode,
        use_judge=False,
        judge_model_name=args.judge_model_name,
        timeout_seconds=args.timeout_seconds,
        target_s=2.0,
    )
    with_judge = benchmark_mode(
        base_url=args.base_url,
        payload=payload,
        sample_size=args.requests_per_mode,
        use_judge=True,
        judge_model_name=args.judge_model_name,
        timeout_seconds=args.timeout_seconds,
        target_s=10.0,
    )

    artifact: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "endpoint": "/review-resume",
        "sample_size": args.requests_per_mode,
        "payload_source": str(args.payload_json),
        "rules_only": rules_only,
        "with_judge": with_judge,
    }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"api_latency_{timestamp}.json"
    out_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=True), encoding="utf-8")

    print("=" * 72)
    print("API LATENCY BENCHMARK COMPLETE")
    print("=" * 72)
    print(f"Artifact: {out_path}")


if __name__ == "__main__":
    main()
