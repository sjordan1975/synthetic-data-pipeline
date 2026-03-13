"""
pytest configuration and fixtures for mini-project-2 tests.

This file contains shared test fixtures and configuration that can be used
across all test modules in this project.
"""

import pytest
from datetime import datetime
from schemas.models import ContactInfo, Education, Experience, Skill, JobRequirements, JobDescription, Resume


@pytest.fixture
def valid_contact_info():
    """Fixture providing a valid ContactInfo instance."""
    return ContactInfo(
        name="John Doe",
        email="john.doe@example.com",
        phone_region="US",
        phone="(202) 555-0173",
        location="San Francisco, CA"
    )


@pytest.fixture
def valid_education():
    """Fixture providing a valid Education instance."""
    return Education(
        institution="Stanford University",
        degree="Bachelor of Science",
        field_of_study="Computer Science",
        graduation_year=2020,
        gpa=3.8
    )


@pytest.fixture
def valid_experience():
    """Fixture providing a valid Experience instance."""
    return Experience(
        company="Tech Corp",
        position="Software Engineer",
        start_date="2020-01-15",
        end_date="2023-01-20",
        description="Developed web applications using React and Node.js",
        achievements=[
            "Reduced page load time by 40%",
            "Led team of 3 developers",
            "Implemented CI/CD pipeline"
        ]
    )


@pytest.fixture
def valid_skills():
    """Fixture providing a list of valid Skill instances."""
    return [
        Skill(name="Python", category="Programming", proficiency="expert"),
        Skill(name="JavaScript", category="Programming", proficiency="expert"),
        Skill(name="React", category="Programming", proficiency="advanced"),
        Skill(name="Docker", category="DevOps", proficiency="intermediate"),
        Skill(name="Spanish", category="Languages", proficiency="advanced")
    ]


@pytest.fixture
def valid_job_requirements():
    """Fixture providing valid JobRequirements instance."""
    return JobRequirements(
        required_skills=["Python", "JavaScript", "React"],
        preferred_skills=["Docker", "AWS", "TypeScript"],
        experience_level="mid_level",
        education_requirements="Bachelor's degree in Computer Science or related field",
        years_experience=3
    )


@pytest.fixture
def valid_resume(valid_contact_info, valid_education, valid_experience, valid_skills):
    """Fixture providing a complete valid Resume instance."""
    return Resume(
        contact_info=valid_contact_info,
        education=[valid_education],
        experience=[valid_experience],
        skills=valid_skills,
        summary="Experienced software engineer with expertise in full-stack development",
        metadata={
            "trace_id": "resume-test-001",
            "generated_at": "2024-01-15T10:00:00Z",
            "prompt_template": "baseline_v1",
            "fit_level": "good",
            "writing_style": "professional",
        },
    )


@pytest.fixture
def sample_invalid_emails():
    """Fixture providing sample invalid email addresses."""
    return [
        "invalid-email",
        "user@",
        "@domain.com",
        "user..name@domain.com",
        "user@domain",
        "user name@domain.com"
    ]


@pytest.fixture
def sample_invalid_phones():
    """Fixture providing sample invalid phone numbers."""
    return [
        "123",
        "555-123",
        "abc-def-ghij",
        "1",
        "12-34"
    ]


@pytest.fixture
def sample_invalid_gpas():
    """Fixture providing sample invalid GPA values."""
    return [
        5.0,    # Too high
        -1.0,   # Negative
        4.1,    # Just above 4.0
        -0.1    # Just below 0.0
    ]


@pytest.fixture
def sample_invalid_dates():
    """Fixture providing sample invalid date formats."""
    return [
        "Jan 2020",      # Not ISO format
        "2020/01/15",    # Wrong separator
        "15-01-2020",    # Wrong format
        "2020-13-01",    # Invalid month
        "2020-01-32"     # Invalid day
    ]


# pytest configuration
pytest_plugins = []

def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    # Add slow marker to tests that might take longer
    for item in items:
        if "property_based" in item.nodeid:
            item.add_marker(pytest.mark.slow)
