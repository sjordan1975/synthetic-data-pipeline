# Advanced Reference

## LLM-as-Judge notes

- Requires `OPENAI_API_KEY` (for example in `mini-projects/.env.local`).
- Slower/costlier than rules-only analysis.
- Additional artifacts when enabled:
  - `outputs/analysis/llm_judge_<timestamp>.jsonl`
  - `outputs/analysis/llm_judge_summary_<timestamp>.json`
  - `outputs/analysis/review_queue_<timestamp>.jsonl`
  - `outputs/analysis/review_queue_summary_<timestamp>.json`

Rate-limit/backoff environment knobs:

- `LLM_RETRY_MAX_ATTEMPTS` (default: `5`)
- `LLM_RETRY_BASE_DELAY_SECONDS` (default: `0.5`)
- `LLM_RETRY_MAX_DELAY_SECONDS` (default: `8.0`)
- `LLM_RETRY_JITTER_RATIO` (default: `0.2`)
- `LLM_BATCH_DELAY_SECONDS` (default: `0.0`)

## Targeted judge evaluation datasets

- `tests/fixtures/adversarial_pairs_v1.jsonl`
- `tests/fixtures/control_pairs_v1.jsonl`

Run:

```bash
python analyzer.py --pairs-jsonl tests/fixtures/adversarial_pairs_v1.jsonl --enable-llm-judge
python analyzer.py --pairs-jsonl tests/fixtures/control_pairs_v1.jsonl --enable-llm-judge
```

## Policy configuration note

- `config/benchmark_policy_v1.yaml` is treated as a versioned specification/reference.
- Runtime fit-band behavior is implemented in `constraint_injector.py` (`FIT_BANDS`).
- Keep policy text and runtime constants synchronized.
