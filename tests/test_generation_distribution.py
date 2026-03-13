"""Distribution validation tests for Phase 3 generated pair artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import generator


def _write_pairs_like_artifact(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _summarize_pairs(path: Path) -> dict:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            rows.append(json.loads(raw))

    total = len(rows)
    resume_template_counts: dict[str, int] = {}
    fit_counts: dict[str, int] = {}
    niche_count = 0

    for row in rows:
        template_id = row["resume"]["metadata"]["prompt_template"]
        fit_level = row["fit_level"]
        is_niche = bool(row["job_description"]["metadata"]["is_niche_role"])

        resume_template_counts[template_id] = resume_template_counts.get(template_id, 0) + 1
        fit_counts[fit_level] = fit_counts.get(fit_level, 0) + 1
        if is_niche:
            niche_count += 1

    return {
        "row_count": total,
        "resume_template_counts": resume_template_counts,
        "fit_counts": fit_counts,
        "niche_count": niche_count,
    }


def test_distribution_validation_for_phase3_volume_and_coverage(tmp_path: Path) -> None:
    plan = generator.build_generation_plan(
        num_jobs=50,
        resumes_per_job=5,
        niche_ratio=0.2,
        seed=42,
    )

    rows = [
        {
            "fit_level": item.fit_level.value,
            "resume": {"metadata": {"prompt_template": item.resume_template_id}},
            "job_description": {"metadata": {"is_niche_role": item.niche_target}},
        }
        for item in plan
    ]

    artifact = tmp_path / "pairs_phase3_distribution.jsonl"
    _write_pairs_like_artifact(artifact, rows)

    summary = _summarize_pairs(artifact)

    assert summary["row_count"] >= 250
    assert set(summary["fit_counts"]) == {fit.value for fit in generator.FIT_LEVEL_SEQUENCE}
    assert summary["niche_count"] > 0

    for template_id, count in summary["resume_template_counts"].items():
        share = count / summary["row_count"]
        assert share <= 0.30, f"Template {template_id} exceeded 30% cap: {share:.2%}"
