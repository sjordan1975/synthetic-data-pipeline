"""Tests for template registry coverage in Phase 3 multi-template setup."""

from __future__ import annotations

from template_loader import TemplateLoader


def test_template_registry_has_required_job_and_resume_template_counts() -> None:
    loader = TemplateLoader()

    job_templates = loader.list_template_ids(kind="job")
    resume_templates = loader.list_template_ids(kind="resume")

    assert len(job_templates) >= 5
    assert len(resume_templates) >= 5


def test_all_active_templates_load_non_empty_content() -> None:
    loader = TemplateLoader()

    template_ids = (
        loader.list_template_ids(kind="job")
        + loader.list_template_ids(kind="resume")
        + loader.list_template_ids(kind="job_requirements")
    )

    for template_id in template_ids:
        text = loader.get_template(template_id)
        assert text.strip()
