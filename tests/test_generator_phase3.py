"""Phase 3 generator planning tests (multi-template + niche coverage)."""

from __future__ import annotations

import generator


def test_build_generation_plan_size_and_niche_ratio() -> None:
    plan = generator.build_generation_plan(
        num_jobs=10,
        resumes_per_job=5,
        niche_ratio=0.3,
        seed=7,
    )

    assert len(plan) == 50

    niche_job_indices = {item.job_index for item in plan if item.niche_target}
    assert len(niche_job_indices) == 3


def test_build_generation_plan_uses_all_templates_when_enough_jobs() -> None:
    plan = generator.build_generation_plan(
        num_jobs=25,
        resumes_per_job=5,
        niche_ratio=0.2,
        seed=11,
    )

    job_templates = {item.job_template_id for item in plan}
    resume_templates = {item.resume_template_id for item in plan}
    fit_levels = {item.fit_level.value for item in plan}

    assert job_templates == set(generator.JOB_TEMPLATE_IDS)
    assert resume_templates == set(generator.RESUME_TEMPLATE_IDS)
    assert fit_levels == {level.value for level in generator.FIT_LEVEL_SEQUENCE}


def test_choose_industry_focus_respects_niche_target() -> None:
    rng = generator.random.Random(1)
    niche_focus = generator.choose_industry_focus(rng=rng, niche_target=True)
    standard_focus = generator.choose_industry_focus(rng=rng, niche_target=False)

    assert niche_focus in generator.NICHE_INDUSTRY_FOCI
    assert standard_focus in generator.STANDARD_INDUSTRY_FOCI
