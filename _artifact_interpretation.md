# Artifact Interpretation Guide

Use this guide to explain what each output means after running validation, analysis, and visualization.

## 1) Validation outputs (Phase 4)

- `outputs/validation/validated_data_<timestamp>.json`
  - Canonical set of records that passed schema validation.
  - Use this as the trusted input for analysis.

- `outputs/validation/invalid_<timestamp>.jsonl`
  - Raw records that failed validation.
  - Useful for debugging generator/correction behavior.

- `outputs/validation/schema_failure_modes_<timestamp>.json`
  - Aggregate counts by schema error type.
  - Answers: "What kind of structural issues are most common?"

Interpretation tip:
- If validation pass rate is low, semantic findings are less reliable until structural quality is fixed.

## 2) Analysis outputs (Phase 5)

- `outputs/analysis/failure_labels_<timestamp>.jsonl`
  - Record-level failure flags and metrics.
  - Answers: "What failed for this specific resume-job pair?"

- `outputs/analysis/failure_breakdown_<timestamp>.json`
  - Grouped failure statistics (for example by fit level and template).
  - Answers: "Where do failures concentrate at segment level?"

- `outputs/analysis/correlation_matrix_<timestamp>.json`
  - Pairwise correlation among failure modes.
  - Answers: "Which failure types co-occur and may share root causes?"

- `outputs/analysis/adjudication_log_<timestamp>.jsonl`
  - Traceability log of analysis/auditing outcomes.

- `outputs/analysis/spot_check_<timestamp>.json`
  - Small sampled summary for sanity checking.

## 3) Visualization outputs (Phase 6): what each chart means

### A) `failure_mode_correlation_heatmap_<timestamp>.png` (+ `.csv`)
- Shows which failure modes move together.
- High positive correlation suggests common upstream causes.

### B) `failure_rates_by_fit_level_<timestamp>.png` (+ `.csv`)
- Compares failure rates across target fit levels.
- Expected pattern: lower rates for `excellent/good`, higher rates for `poor/complete_mismatch`.

### C) `failure_rates_by_template_<timestamp>.png` (+ `.csv`)
- Compares failure rates by resume template.
- Helps identify prompt templates that are systematically underperforming.

### D) `niche_vs_standard_failure_<timestamp>.png` (+ `.csv`)
- Compares failure rates for niche vs standard roles.
- Useful for checking whether specialization introduces disproportionate failure risk.

### E) `schema_error_distribution_<timestamp>.png` (+ `.csv`)
- Visual summary of schema validation error counts.
- Helps prioritize highest-impact structural fixes first.

### F) `interpretation_notes_<timestamp>.md`
- Auto-generated narrative summary of key chart findings.
- Good starting point for reporting or stakeholder updates.
