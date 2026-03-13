"""Tests for Experience date handling with typed date fields."""

from datetime import date

import pytest
from pydantic import ValidationError

from schemas.models import Experience


def test_experience_dates_parse_to_date_objects() -> None:
    exp = Experience(
        company="Tech Corp",
        position="Software Engineer",
        start_date="2020-01-15",
        end_date="2023-01-20",
        description="Developed web applications using React and Node.js",
    )

    assert isinstance(exp.start_date, date)
    assert isinstance(exp.end_date, date)
    assert exp.start_date == date(2020, 1, 15)
    assert exp.end_date == date(2023, 1, 20)


def test_experience_end_date_optional_for_current_role() -> None:
    exp = Experience(
        company="Tech Corp",
        position="Software Engineer",
        start_date="2020-01-15",
        end_date=None,
        description="Developed web applications using React and Node.js",
    )

    assert exp.end_date is None


def test_experience_end_date_must_be_after_start_date() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Experience(
            company="Tech Corp",
            position="Software Engineer",
            start_date="2023-01-20",
            end_date="2020-01-15",
            description="Developed web applications using React and Node.js",
        )

    assert "end_date must be after start_date" in str(exc_info.value)


def test_experience_invalid_date_format_rejected() -> None:
    with pytest.raises(ValidationError):
        Experience(
            company="Tech Corp",
            position="Software Engineer",
            start_date="Jan 2020",
            end_date="2023-01-20",
            description="Developed web applications using React and Node.js",
        )
