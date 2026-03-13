# Runtime Metrics, Benchmarking, and Summary

This guide focuses on runtime evidence artifacts (API latency, stage timings, and metrics report).

## Generate required pipeline summary artifact

After running generation, validation, and analysis (and optional correction),
generate the required summary deliverable:

```bash
python summarize_pipeline.py
```

This writes:

- `outputs/pipeline_summary_<timestamp>.json`

## Optional enrichments for pipeline summary

1) Include explicit stage durations in the summary:

```bash
python summarize_pipeline.py --stage-times-json outputs/stage_times.json
```

2) Include API latency metrics automatically:

- If `outputs/metrics/api_latency_<timestamp>.json` exists, `summarize_pipeline.py`
  automatically ingests the latest file and stores it under `summary.api_latency`.

## Capture runtime metrics artifacts (optional)

API latency benchmark (`/review-resume`, rules-only vs with-judge):

```bash
# Start API first in another terminal: uvicorn api:app --reload --port 8000
python benchmark_api.py --requests-per-mode 10
```

This writes:

- `outputs/metrics/api_latency_<timestamp>.json`

Capture stage runtime while running each phase command (repeat per phase):

```bash
python capture_stage_time.py --stage generation --times-json outputs/stage_times.json -- \
  python generator.py --num-jobs 50 --resumes-per-job 5 --output-dir data/generated
```

Then pass stage timings into summary generation (optional):

```bash
python summarize_pipeline.py --stage-times-json outputs/stage_times.json
```

## Optional derived metrics report

After producing one or more runtime metrics artifacts, generate a concise
markdown summary:

- `outputs/pipeline_summary_<timestamp>.json`
- `outputs/metrics/api_latency_<timestamp>.json` (if API benchmark was run)
- `outputs/stage_times.json` (if stage timing capture was used)

Run:

```bash
python metrics_report.py
```

Output:

- `outputs/metrics/metrics_report_<timestamp>.md`
