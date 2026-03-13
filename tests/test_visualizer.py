"""Targeted tests for Phase 6 visualizer data-shaping and artifact discovery."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import pytest

import visualizer


def test_find_latest_file_returns_most_recent_match(tmp_path: Path) -> None:
    first = tmp_path / "failure_labels_20260101_000001.jsonl"
    second = tmp_path / "failure_labels_20260101_000002.jsonl"

    first.write_text("{}\n", encoding="utf-8")
    time.sleep(0.01)
    second.write_text("{}\n", encoding="utf-8")

    latest = visualizer.find_latest_file(tmp_path, "failure_labels_*.jsonl")
    assert latest == second


def test_build_fit_level_rates_df_orders_levels_and_extracts_rates() -> None:
    payload = {
        "by_fit_level": {
            "poor": {
                "count": 5,
                "mean_overlap": 0.2,
                "mean_failure_count": 1.2,
                "experience_mismatch_rate": 0.1,
                "seniority_mismatch_rate": 0.2,
                "missing_core_skills_rate": 0.9,
                "hallucinated_skills_rate": 0.0,
                "awkward_language_rate": 0.3,
            },
            "excellent": {
                "count": 5,
                "mean_overlap": 0.9,
                "mean_failure_count": 0.0,
                "experience_mismatch_rate": 0.0,
                "seniority_mismatch_rate": 0.0,
                "missing_core_skills_rate": 0.0,
                "hallucinated_skills_rate": 0.0,
                "awkward_language_rate": 0.0,
            },
        }
    }

    df = visualizer.build_fit_level_rates_df(payload)

    assert list(df["fit_level"].astype(str)) == ["excellent", "poor"]
    assert "missing_core_skills_rate" in df.columns
    assert df.loc[df["fit_level"].astype(str) == "poor", "missing_core_skills_rate"].iloc[0] == 0.9


def test_build_niche_vs_standard_df_computes_group_rates() -> None:
    failure_labels_df = pd.DataFrame(
        [
            {
                "trace_id": "a",
                "skills_overlap": 0.8,
                "failure_count": 0,
                "experience_mismatch": False,
                "seniority_mismatch": False,
                "missing_core_skills": False,
                "hallucinated_skills": False,
                "awkward_language": False,
            },
            {
                "trace_id": "b",
                "skills_overlap": 0.2,
                "failure_count": 2,
                "experience_mismatch": True,
                "seniority_mismatch": True,
                "missing_core_skills": True,
                "hallucinated_skills": False,
                "awkward_language": False,
            },
        ]
    )
    niche_mapping_df = pd.DataFrame(
        [
            {"trace_id": "a", "is_niche_role": False},
            {"trace_id": "b", "is_niche_role": True},
        ]
    )

    df = visualizer.build_niche_vs_standard_df(failure_labels_df, niche_mapping_df)

    assert set(df["role_type"]) == {"standard", "niche"}
    niche_row = df[df["role_type"] == "niche"].iloc[0]
    assert niche_row["mean_failure_count"] == 2.0
    assert niche_row["missing_core_skills_rate"] == 1.0


def test_load_artifacts_requires_expected_files(tmp_path: Path) -> None:
    analysis_dir = tmp_path / "analysis"
    validation_dir = tmp_path / "validation"
    analysis_dir.mkdir()
    validation_dir.mkdir()

    (analysis_dir / "failure_labels_20260101_000000.jsonl").write_text("{}\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        visualizer.load_artifacts(
            analysis_dir=analysis_dir,
            validation_dir=validation_dir,
            validated_data_json=None,
        )


def test_run_visualization_writes_png_csv_and_notes(tmp_path: Path) -> None:
    analysis_dir = tmp_path / "analysis"
    validation_dir = tmp_path / "validation"
    output_dir = tmp_path / "visualizations"
    analysis_dir.mkdir()
    validation_dir.mkdir()

    (analysis_dir / "failure_labels_20260101_000000.jsonl").write_text(
        json.dumps(
            {
                "trace_id": "t1",
                "generated_fit_label": "good",
                "computed_fit_label": "good",
                "skills_overlap": 0.7,
                "experience_mismatch": False,
                "seniority_mismatch": False,
                "missing_core_skills": False,
                "hallucinated_skills": False,
                "awkward_language": False,
                "failure_count": 0,
                "resume_template": "resume_professional_v1",
                "requires_human_review": False,
                "review_reason_codes": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    (analysis_dir / "correlation_matrix_20260101_000000.json").write_text(
        json.dumps(
            {
                "matrix": {
                    "skills_overlap": {
                        "skills_overlap": 1.0,
                        "experience_mismatch": 0.0,
                        "seniority_mismatch": 0.0,
                        "missing_core_skills": -0.5,
                        "hallucinated_skills": 0.0,
                        "awkward_language": 0.0,
                    },
                    "experience_mismatch": {
                        "skills_overlap": 0.0,
                        "experience_mismatch": 1.0,
                        "seniority_mismatch": 0.0,
                        "missing_core_skills": 0.0,
                        "hallucinated_skills": 0.0,
                        "awkward_language": 0.0,
                    },
                    "seniority_mismatch": {
                        "skills_overlap": 0.0,
                        "experience_mismatch": 0.0,
                        "seniority_mismatch": 1.0,
                        "missing_core_skills": 0.0,
                        "hallucinated_skills": 0.0,
                        "awkward_language": 0.0,
                    },
                    "missing_core_skills": {
                        "skills_overlap": -0.5,
                        "experience_mismatch": 0.0,
                        "seniority_mismatch": 0.0,
                        "missing_core_skills": 1.0,
                        "hallucinated_skills": 0.0,
                        "awkward_language": 0.0,
                    },
                    "hallucinated_skills": {
                        "skills_overlap": 0.0,
                        "experience_mismatch": 0.0,
                        "seniority_mismatch": 0.0,
                        "missing_core_skills": 0.0,
                        "hallucinated_skills": 1.0,
                        "awkward_language": 0.0,
                    },
                    "awkward_language": {
                        "skills_overlap": 0.0,
                        "experience_mismatch": 0.0,
                        "seniority_mismatch": 0.0,
                        "missing_core_skills": 0.0,
                        "hallucinated_skills": 0.0,
                        "awkward_language": 1.0,
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    (analysis_dir / "failure_breakdown_20260101_000000.json").write_text(
        json.dumps(
            {
                "by_fit_level": {
                    "good": {
                        "count": 1,
                        "mean_overlap": 0.7,
                        "mean_failure_count": 0.0,
                        "experience_mismatch_rate": 0.0,
                        "seniority_mismatch_rate": 0.0,
                        "missing_core_skills_rate": 0.0,
                        "hallucinated_skills_rate": 0.0,
                        "awkward_language_rate": 0.0,
                    }
                },
                "by_template": {
                    "resume_professional_v1": {
                        "count": 1,
                        "mean_overlap": 0.7,
                        "mean_failure_count": 0.0,
                        "awkward_language_rate": 0.0,
                        "hallucinated_skills_rate": 0.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    (validation_dir / "schema_failure_modes_20260101_000000.json").write_text(
        json.dumps(
            {
                "error_type_distribution": {
                    "missing_required_fields": 0,
                    "type_mismatches": 0,
                    "format_violations": 0,
                    "logical_inconsistencies": 0,
                    "enum_mapping_failures": 0,
                }
            }
        ),
        encoding="utf-8",
    )

    (validation_dir / "validated_data_20260101_000000.json").write_text(
        json.dumps(
            {
                "records": {
                    "pairs": [
                        {
                            "trace_id": "t1",
                            "resume": {
                                "contact_info": {
                                    "name": "John Doe",
                                    "email": "john@example.com",
                                    "phone_region": "US",
                                    "phone": "+12025550173",
                                    "location": "SF, CA",
                                },
                                "education": [
                                    {
                                        "institution": "Uni",
                                        "degree": "BS",
                                        "field_of_study": "CS",
                                        "graduation_year": 2020,
                                        "gpa": 3.5,
                                    }
                                ],
                                "experience": [
                                    {
                                        "company": "Co",
                                        "position": "Engineer",
                                        "start_date": "2021-01-01",
                                        "end_date": "2023-01-01",
                                        "description": "Built APIs for customers.",
                                        "achievements": ["Improved latency"],
                                    }
                                ],
                                "skills": [
                                    {"name": "Python", "category": "Programming", "proficiency": "expert"},
                                    {"name": "SQL", "category": "Programming", "proficiency": "advanced"},
                                    {"name": "Docker", "category": "DevOps", "proficiency": "intermediate"},
                                ],
                                "metadata": {
                                    "trace_id": "resume-1",
                                    "generated_at": "2024-01-01T00:00:00Z",
                                    "prompt_template": "resume_professional_v1",
                                    "fit_level": "good",
                                    "writing_style": "professional",
                                    "used_fallback": False,
                                },
                            },
                            "job_description": {
                                "company": "Company",
                                "industry": "Software",
                                "company_size": "51-200",
                                "position": "Engineer",
                                "location": "SF, CA",
                                "description": "Looking for engineer with Python and SQL background in backend APIs.",
                                "requirements": {
                                    "required_skills": ["Python", "SQL"],
                                    "preferred_skills": ["Docker"],
                                    "experience_level": "mid_level",
                                    "education_requirements": "Bachelor's degree",
                                    "years_experience": 2,
                                },
                                "metadata": {
                                    "trace_id": "job-1",
                                    "generated_at": "2024-01-01T00:00:00Z",
                                    "is_niche_role": False,
                                    "used_fallback": False,
                                },
                            },
                            "fit_level": "good",
                            "analysis_date": "2024-01-01T00:00:00Z",
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    paths = visualizer.run_visualization(
        analysis_dir=analysis_dir,
        validation_dir=validation_dir,
        output_dir=output_dir,
        validated_data_json=None,
    )

    assert all(path.exists() for path in paths.values())
    png_count = len(list(output_dir.glob("*.png")))
    csv_count = len(list(output_dir.glob("*.csv")))
    assert png_count >= 5
    assert csv_count >= 5
