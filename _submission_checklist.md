# Grader Handoff Checklist (submission-ready)

Use this as the final artifact checklist before submitting.

## Required artifacts

1) Generated data (`data/generated/`)
- `jobs_<timestamp>.jsonl`
- `resumes_<timestamp>.jsonl`
- `pairs_<timestamp>.jsonl`

2) Validation outputs (`outputs/validation/`)
- `validated_data_<timestamp>.json`
- `invalid_<timestamp>.jsonl`
- `schema_failure_modes_<timestamp>.json`

3) Failure analysis outputs (`outputs/analysis/`)
- `failure_labels_<timestamp>.jsonl`
- Failure statistics evidence (rates/correlations/distributions), for example:
  - `failure_breakdown_<timestamp>.json`
  - `correlation_matrix_<timestamp>.json`

4) Visualizations (PNG)
- Failure mode correlation matrix
- Failure rates by fit level
- Failure rates by template
- Niche vs standard roles
- Schema validation heatmap

5) API requirements
- FastAPI app runs and serves:
  - `POST /review-resume`
  - `GET /health`
  - `GET /analysis/failure-rates`
  - `/docs`

6) Pipeline summary (`outputs/`)
- `pipeline_summary_<timestamp>.json` including:
  - total records generated
  - validation success rate
  - failure mode distribution
  - correction success rate (if correction run)
  - processing time per stage

## Optional supporting evidence (recommended)

- `outputs/metrics/api_latency_<timestamp>.json`
- `outputs/stage_times.json`
- `outputs/metrics/metrics_report_<timestamp>.md`
- `outputs/visualizations/interpretation_notes_<timestamp>.md`

---

## Presentation-Ready Summary (1 line per artifact)

- `validated_data_<timestamp>.json`: This is the trusted, schema-valid dataset used for all downstream analysis.
- `invalid_<timestamp>.jsonl`: These are records that failed schema checks and need generator/correction attention.
- `schema_failure_modes_<timestamp>.json`: This shows which structural error types are most common and where to prioritize fixes.
- `failure_labels_<timestamp>.jsonl`: This gives record-level semantic failure reasons for each resume-job pair.
- `failure_breakdown_<timestamp>.json`: This summarizes where failures concentrate by segment (for example fit level/template).
- `correlation_matrix_<timestamp>.json`: This identifies failure modes that co-occur and may share root causes.
- `adjudication_log_<timestamp>.jsonl`: This is the audit trail for analysis decisions and review routing outcomes.
- `spot_check_<timestamp>.json`: This is a quick sanity sample to verify outputs look reasonable.
- `failure_mode_correlation_heatmap_<timestamp>.png`: This visualizes which failure modes move together most strongly.
- `failure_rates_by_fit_level_<timestamp>.png`: This checks whether failure rates trend as expected across fit levels.
- `failure_rates_by_template_<timestamp>.png`: This surfaces prompt templates that systematically underperform.
- `niche_vs_standard_failure_<timestamp>.png`: This shows whether niche roles have disproportionate failure risk.
- `schema_error_distribution_<timestamp>.png`: This highlights the dominant schema error classes for fast remediation.
- `interpretation_notes_<timestamp>.md`: This provides an auto-written narrative of the most important chart-level findings.