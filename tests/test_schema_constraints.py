"""Tests for schema constraints not covered by contact/date-focused tests."""

import pytest
from pydantic import ValidationError

from schemas.models import (
    ContactInfo,
    Education,
    JobDescription,
    JobMetadata,
    JobRequirements,
    Resume,
    ResumeJobPair,
    Skill,
)


def build_valid_job_metadata(**overrides):
    """Build a valid JobMetadata instance with optional overrides."""
    data = {
        "trace_id": "job-test-001",
        "generated_at": "2024-01-15T10:00:00Z",
        "is_niche_role": False,
    }
    data.update(overrides)
    return JobMetadata(**data)


def build_valid_job_description(valid_job_requirements, **overrides):
    """Build a valid JobDescription instance with optional overrides."""
    data = {
        "company": "StartupXYZ",
        "industry": "Software",
        "company_size": "51-200",
        "position": "Software Engineer",
        "location": "San Francisco, CA",
        "description": "We are looking for an experienced engineer to build APIs and internal tools.",
        "requirements": valid_job_requirements,
        "metadata": build_valid_job_metadata(),
        "salary_range": "120k-160k",
    }
    data.update(overrides)
    return JobDescription(**data)


def test_phone_region_normalized_to_uppercase() -> None:
    contact = ContactInfo(
        name="Test User",
        email="test@example.com",
        phone_region="us",
        phone="2025550173",
        location="San Francisco, CA",
    )

    assert contact.phone_region == "US"


@pytest.mark.parametrize("phone_region", ["U", "USA", "U1"])
def test_phone_region_rejects_invalid_codes(phone_region: str) -> None:
    with pytest.raises(ValidationError) as exc_info:
        ContactInfo(
            name="Test User",
            email="test@example.com",
            phone_region=phone_region,
            phone="+12025550173",
            location="San Francisco, CA",
        )

    assert "phone_region must be a 2-letter ISO country code" in str(exc_info.value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("graduation_year", 1949),
        ("graduation_year", 2031),
        ("gpa", -0.1),
        ("gpa", 4.1),
    ],
)
def test_education_numeric_bounds(field: str, value: float | int) -> None:
    payload = {
        "institution": "Stanford University",
        "degree": "Bachelor of Science",
        "field_of_study": "Computer Science",
        "graduation_year": 2020,
        "gpa": 3.5,
    }
    payload[field] = value

    with pytest.raises(ValidationError):
        Education(**payload)


@pytest.mark.parametrize("proficiency", ["beginner", "intermediate", "advanced", "expert"])
def test_skill_proficiency_accepts_enum_values(proficiency: str) -> None:
    skill = Skill(name="Python", category="Programming", proficiency=proficiency)
    assert skill.proficiency.value == proficiency


def test_skill_proficiency_rejects_unknown_value() -> None:
    with pytest.raises(ValidationError):
        Skill(name="Python", category="Programming", proficiency="guru")


def test_job_requirements_required_skills_must_not_be_empty() -> None:
    with pytest.raises(ValidationError):
        JobRequirements(
            required_skills=[],
            preferred_skills=["Docker"],
            experience_level="mid_level",
            education_requirements="Bachelor's degree in Computer Science",
            years_experience=3,
        )


def test_job_requirements_experience_level_rejects_unknown_value() -> None:
    with pytest.raises(ValidationError):
        JobRequirements(
            required_skills=["Python"],
            preferred_skills=["Docker"],
            experience_level="lead",
            education_requirements="Bachelor's degree in Computer Science",
            years_experience=3,
        )


@pytest.mark.parametrize("years", [-1, 31])
def test_job_requirements_years_experience_bounds(years: int) -> None:
    with pytest.raises(ValidationError):
        JobRequirements(
            required_skills=["Python"],
            preferred_skills=["Docker"],
            experience_level="mid_level",
            education_requirements="Bachelor's degree in Computer Science",
            years_experience=years,
        )


def test_resume_requires_at_least_one_education(valid_contact_info, valid_experience, valid_skills) -> None:
    with pytest.raises(ValidationError):
        Resume(
            contact_info=valid_contact_info,
            education=[],
            experience=[valid_experience],
            skills=valid_skills[:3],
        )


def test_resume_requires_at_least_one_experience(valid_contact_info, valid_education, valid_skills) -> None:
    with pytest.raises(ValidationError):
        Resume(
            contact_info=valid_contact_info,
            education=[valid_education],
            experience=[],
            skills=valid_skills[:3],
        )


def test_resume_requires_at_least_three_skills(valid_contact_info, valid_education, valid_experience, valid_skills) -> None:
    with pytest.raises(ValidationError):
        Resume(
            contact_info=valid_contact_info,
            education=[valid_education],
            experience=[valid_experience],
            skills=valid_skills[:2],
        )


def test_resume_summary_min_length_enforced(valid_contact_info, valid_education, valid_experience, valid_skills) -> None:
    with pytest.raises(ValidationError):
        Resume(
            contact_info=valid_contact_info,
            education=[valid_education],
            experience=[valid_experience],
            skills=valid_skills[:3],
            summary="Too short",
        )


def test_job_metadata_trace_id_min_length() -> None:
    with pytest.raises(ValidationError):
        JobMetadata(trace_id="ab")


def test_job_metadata_generated_at_must_be_datetime() -> None:
    with pytest.raises(ValidationError):
        JobMetadata(generated_at="not-a-date")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("industry", "A"),
        ("company_size", "X"),
    ],
)
def test_job_description_company_fields_min_length(valid_job_requirements, field: str, value: str) -> None:
    with pytest.raises(ValidationError):
        build_valid_job_description(valid_job_requirements, **{field: value})


def test_job_description_description_min_length(valid_job_requirements) -> None:
    with pytest.raises(ValidationError):
        build_valid_job_description(valid_job_requirements, description="Too short")


def test_job_description_salary_range_min_length(valid_job_requirements) -> None:
    with pytest.raises(ValidationError):
        build_valid_job_description(valid_job_requirements, salary_range="1-2")


def test_resume_job_pair_fit_level_accepts_enum_values(valid_resume, valid_job_requirements) -> None:
    pair = ResumeJobPair(
        resume=valid_resume,
        job_description=build_valid_job_description(valid_job_requirements),
        fit_level="good",
        trace_id="pair-001",
    )

    assert pair.fit_level.value == "good"


def test_resume_job_pair_fit_level_rejects_unknown_value(valid_resume, valid_job_requirements) -> None:
    with pytest.raises(ValidationError):
        ResumeJobPair(
            resume=valid_resume,
            job_description=build_valid_job_description(valid_job_requirements),
            fit_level="amazing",
            trace_id="pair-001",
        )


def test_resume_job_pair_trace_id_min_length(valid_resume, valid_job_requirements) -> None:
    with pytest.raises(ValidationError):
        ResumeJobPair(
            resume=valid_resume,
            job_description=build_valid_job_description(valid_job_requirements),
            fit_level="good",
            trace_id="ab",
        )
