# Stage Runbook

Detailed phase-by-phase commands for running the pipeline.

## 1) Generation

```bash
python generator.py \
  --num-jobs 50 \
  --resumes-per-job 5 \
  --niche-ratio 0.2 \
  --seed 42 \
  --output-dir data/generated
```

Expected volume: `50 x 5 = 250` resume-job pairs.

Artifacts:
- `data/generated/jobs_<timestamp>.jsonl`
- `data/generated/resumes_<timestamp>.jsonl`
- `data/generated/pairs_<timestamp>.jsonl`

## 2) Validation

```bash
python validator.py \
  --input-dir data/generated \
  --output-dir outputs/validation
```

Artifacts include validated/invalid partitions and summary metrics under `outputs/validation/`.

## 3) Analysis

Rules-only baseline:

```bash
python analyzer.py \
  --validated-data-json outputs/validation/validated_data_<timestamp>.json \
  --output-dir outputs/analysis
```

Optional LLM-as-Judge:

```bash
python analyzer.py \
  --validated-data-json outputs/validation/validated_data_<timestamp>.json \
  --output-dir outputs/analysis \
  --enable-llm-judge
```

Optional model override:

```bash
python analyzer.py --enable-llm-judge --judge-model-name gpt-4o-mini
```

## 4) Correction (optional)

```bash
python corrector.py \
  --validation-dir outputs/validation \
  --output-dir outputs/corrections
```

Use this when you want to attempt automatic fixes for invalid records and re-check correction success rates.

## 5) API exposure (FastAPI)

```bash
uvicorn api:app --reload --port 8000
```

Endpoints:
- `POST /review-resume` (optional `?use_judge=true`)
- `GET /health`
- `GET /analysis/failure-rates`

Sample requests:

```bash
curl -s http://localhost:8000/health

curl -s -X POST "http://localhost:8000/review-resume" \
  -H "Content-Type: application/json" \
  -d @tests/fixtures/control_pairs_v1_first_record.json

curl -s -X POST "http://localhost:8000/review-resume?use_judge=true" \
  -H "Content-Type: application/json" \
  -d @tests/fixtures/control_pairs_v1_first_record.json

curl -s "http://localhost:8000/analysis/failure-rates"
```

## Verification and tests

Generation coverage checks:

```bash
pytest -q tests/test_generator_phase3.py tests/test_template_registry.py tests/test_generation_distribution.py
```

API checks:

```bash
pytest -q tests/test_api.py
```

Or run the full suite:

```bash
pytest -q
```

## Optional visualization

```bash
python visualizer.py
```

Optional explicit inputs:

```bash
python visualizer.py \
  --analysis-dir outputs/analysis \
  --validation-dir outputs/validation \
  --validated-data-json outputs/validation/validated_data_<timestamp>.json \
  --output-dir outputs/visualizations
```
