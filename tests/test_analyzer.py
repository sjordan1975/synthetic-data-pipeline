"""Targeted unit tests for Phase 5 analyzer metrics and outputs."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import analyzer
from schemas.models import Experience, JobDescription, Resume, ResumeJobPair, Skill


class StubJudgeClient:
    """Deterministic fake judge client for analyzer tests."""

    def __init__(self, payloads: list[dict]) -> None:
        self._payloads = payloads
        self.calls = 0

    def evaluate_pair(self, pair: ResumeJobPair, rule_analysis: analyzer.PairAnalysis) -> dict:
        self.calls += 1
        if self._payloads:
            return self._payloads.pop(0)
        return {
            "has_hallucinations": False,
            "hallucination_details": "",
            "has_awkward_language": False,
            "awkward_language_details": "",
            "overall_quality_score": 0.8,
            "fit_assessment": "Good match.",
            "recommendations": [],
            "red_flags": [],
        }


def build_job(valid_job_requirements, **requirement_overrides) -> JobDescription:
    """Build a valid job description with optional requirement overrides."""
    requirements_payload = valid_job_requirements.model_dump(mode="json")
    requirements_payload.update(requirement_overrides)

    return JobDescription(
        company="StartupXYZ",
        industry="Software",
        company_size="51-200",
        position="Software Engineer",
        location="San Francisco, CA",
        description="We are looking for an experienced engineer to build APIs and internal tools.",
        requirements=requirements_payload,
        metadata={
            "trace_id": "job-test-001",
            "generated_at": "2024-01-15T10:00:00Z",
            "is_niche_role": False,
            "used_fallback": False,
        },
    )


def build_pair(valid_resume, valid_job_requirements, trace_id: str = "pair-test-001") -> ResumeJobPair:
    """Build a baseline valid pair for analyzer tests."""
    job = build_job(valid_job_requirements)
    return ResumeJobPair(
        resume=valid_resume,
        job_description=job,
        fit_level="good",
        trace_id=trace_id,
    )


def test_normalize_skill_applies_expected_reductions() -> None:
    assert analyzer.normalize_skill("Senior Python Developer 3.11") == "senior python"
    assert analyzer.normalize_skill("Node.js Engineer") == "node"


def test_jaccard_overlap_simple_case() -> None:
    resume = {"python", "sql", "docker"}
    required = {"python", "sql", "aws"}
    assert analyzer.jaccard_overlap(resume, required) == 0.5


def test_count_buzzwords_includes_move_the_needle_phrase() -> None:
    text = "We move the needle on outcomes and drive synergy across teams."
    assert analyzer.count_buzzwords(text) >= 2


def test_has_local_token_repetition_detects_same_word_close_proximity() -> None:
    text = "We created synergy synergy synergy across cross functional delivery streams."
    assert analyzer.has_local_token_repetition(text) is True


def test_analyze_pair_flags_seniority_mismatch(valid_resume, valid_job_requirements) -> None:
    pair = build_pair(valid_resume, valid_job_requirements, trace_id="pair-exp-senior")

    # Force a large level gap between inferred resume seniority and job requirement.
    pair.job_description.requirements.experience_level = "entry_level"
    pair.resume.experience = [
        Experience(
            company="Big Tech",
            position="Senior Software Engineer",
            start_date=date(2015, 1, 1),
            end_date=date(2023, 1, 1),
            description="Led backend architecture and mentored engineers.",
            achievements=["Shipped large-scale platform migration."],
        )
    ]

    result = analyzer.analyze_pair(pair)

    assert result.experience_mismatch is False
    assert result.seniority_mismatch is True


def test_analyze_pair_flags_experience_mismatch(valid_resume, valid_job_requirements) -> None:
    pair = build_pair(valid_resume, valid_job_requirements, trace_id="pair-exp-mismatch")
    pair.job_description.requirements.years_experience = 10
    pair.resume.experience = [
        Experience(
            company="Intern Co",
            position="Intern",
            start_date=date(2025, 1, 1),
            end_date=date(2025, 8, 1),
            description="Supported engineering team.",
            achievements=["Completed assigned tasks."],
        )
    ]

    result = analyzer.analyze_pair(pair)
    assert result.experience_mismatch is True


def test_analyze_pair_flags_missing_core_skills(valid_resume, valid_job_requirements) -> None:
    pair = build_pair(valid_resume, valid_job_requirements, trace_id="pair-core-skills")
    pair.resume.skills = [
        Skill(name="Docker", category="DevOps", proficiency="advanced"),
        Skill(name="Kubernetes", category="DevOps", proficiency="advanced"),
        Skill(name="Terraform", category="DevOps", proficiency="intermediate"),
    ]

    result = analyzer.analyze_pair(pair)
    assert result.missing_core_skills is True


def test_analyze_pair_flags_hallucinated_skills_for_entry_profile(valid_resume, valid_job_requirements) -> None:
    pair = build_pair(valid_resume, valid_job_requirements, trace_id="pair-hallucinated")

    pair.resume.experience = [
        Experience(
            company="Intern Co",
            position="Intern",
            start_date=date(2025, 1, 1),
            end_date=date(2025, 12, 1),
            description="Assisted engineering team on basic tasks.",
            achievements=["Completed internship projects successfully."],
        )
    ]
    pair.resume.skills = [
        Skill(name=f"Skill {idx}", category="General", proficiency="expert") for idx in range(12)
    ]

    result = analyzer.analyze_pair(pair)
    assert result.hallucinated_skills is True


def test_analyze_pair_flags_awkward_language(valid_resume, valid_job_requirements) -> None:
    pair = build_pair(valid_resume, valid_job_requirements, trace_id="pair-awkward")
    pair.resume.summary = " ".join(["synergy"] * 8) + " results-driven dynamic innovative"

    result = analyzer.analyze_pair(pair)
    assert result.awkward_language is True


def test_build_judge_prompt_includes_explicit_rubric(valid_resume, valid_job_requirements) -> None:
    pair = build_pair(valid_resume, valid_job_requirements, trace_id="pair-judge-prompt")
    rule_analysis = analyzer.analyze_pair(pair)

    prompt = analyzer.build_judge_prompt(pair, rule_analysis)

    assert "1) Hallucinations" in prompt
    assert "2) Awkward Language" in prompt
    assert "3) Fit Assessment" in prompt
    assert "4) Red Flags" in prompt
    assert "Positive examples (flag):" in prompt
    assert "Negative examples (do not auto-flag):" in prompt
    assert "cite 1-3 concrete snippets or facts" in prompt


def test_run_analysis_writes_phase5_artifacts(tmp_path: Path, valid_resume, valid_job_requirements) -> None:
    pair = build_pair(valid_resume, valid_job_requirements, trace_id="pair-run-analysis")

    pairs_path = tmp_path / "pairs_test.jsonl"
    pairs_path.write_text(pair.model_dump_json() + "\n", encoding="utf-8")

    output_paths = analyzer.run_analysis(
        input_pairs_jsonl=pairs_path,
        validated_data_json=None,
        output_dir=tmp_path / "analysis",
        seed=42,
    )

    assert output_paths["failure_labels"].exists()
    assert output_paths["correlation_matrix"].exists()
    assert output_paths["failure_breakdown"].exists()
    assert output_paths["adjudication_log"].exists()
    assert output_paths["spot_check"].exists()

    labels = [json.loads(line) for line in output_paths["failure_labels"].read_text(encoding="utf-8").splitlines()]
    assert len(labels) == 1
    assert labels[0]["trace_id"] == "pair-run-analysis"


def test_analyze_pairs_with_judge_adds_review_reason(valid_resume, valid_job_requirements) -> None:
    pair = build_pair(valid_resume, valid_job_requirements, trace_id="pair-judge-flagged")
    judge = StubJudgeClient(
        payloads=[
            {
                "has_hallucinations": True,
                "hallucination_details": "Claimed impossible timeline.",
                "has_awkward_language": False,
                "awkward_language_details": "",
                "overall_quality_score": 0.35,
                "fit_assessment": "Low confidence in authenticity.",
                "recommendations": ["Verify employment history"],
                "red_flags": ["Timeline inconsistency"],
            }
        ]
    )

    labels, adjudication, judge_rows = analyzer.analyze_pairs([pair], judge_client=judge)

    assert judge.calls == 1
    assert labels[0]["requires_human_review"] is True
    assert "llm_judge_flagged" in labels[0]["review_reason_codes"]
    assert adjudication[0]["adjudicator_type"] == "human_review"
    assert judge_rows[0]["has_hallucinations"] is True
    assert judge_rows[0]["judge_error"] is None


def test_analyze_pairs_with_recommendations_and_borderline_score_triggers_review(
    valid_resume, valid_job_requirements
) -> None:
    pair = build_pair(valid_resume, valid_job_requirements, trace_id="pair-judge-recommendation")
    judge = StubJudgeClient(
        payloads=[
            {
                "has_hallucinations": False,
                "hallucination_details": "No hard hallucination evidence.",
                "has_awkward_language": False,
                "awkward_language_details": "Mostly readable.",
                "overall_quality_score": 0.65,
                "fit_assessment": "Reasonable fit with some uncertainty.",
                "recommendations": ["Verify scope of migration ownership"],
                "red_flags": [],
            }
        ]
    )

    labels, adjudication, judge_rows = analyzer.analyze_pairs([pair], judge_client=judge)

    assert labels[0]["requires_human_review"] is True
    assert "llm_judge_recommendation" in labels[0]["review_reason_codes"]
    assert adjudication[0]["adjudicator_type"] == "human_review"
    assert judge_rows[0]["recommendations"] == ["Verify scope of migration ownership"]


def test_analyze_pairs_with_invalid_judge_output_flags_error(valid_resume, valid_job_requirements) -> None:
    pair = build_pair(valid_resume, valid_job_requirements, trace_id="pair-judge-invalid")
    judge = StubJudgeClient(
        payloads=[
            {
                "has_hallucinations": False,
                # Missing required fields on purpose.
                "overall_quality_score": 0.8,
            }
        ]
    )

    labels, _, judge_rows = analyzer.analyze_pairs([pair], judge_client=judge)

    assert labels[0]["requires_human_review"] is True
    assert "llm_judge_error" in labels[0]["review_reason_codes"]
    assert judge_rows[0]["judge_error"] is not None


def test_run_analysis_with_llm_judge_writes_judge_artifacts(tmp_path: Path, valid_resume, valid_job_requirements) -> None:
    pair = build_pair(valid_resume, valid_job_requirements, trace_id="pair-run-analysis-judge")
    pairs_path = tmp_path / "pairs_judge_test.jsonl"
    pairs_path.write_text(pair.model_dump_json() + "\n", encoding="utf-8")

    class LocalJudge(StubJudgeClient):
        pass

    original_client = analyzer.OpenAIJudgeClient
    analyzer.OpenAIJudgeClient = lambda model_name: LocalJudge(  # type: ignore[assignment]
        payloads=[
            {
                "has_hallucinations": False,
                "hallucination_details": "",
                "has_awkward_language": False,
                "awkward_language_details": "",
                "overall_quality_score": 0.87,
                "fit_assessment": "Strong overall fit.",
                "recommendations": ["Minor wording cleanup"],
                "red_flags": [],
            }
        ]
    )
    try:
        output_paths = analyzer.run_analysis(
            input_pairs_jsonl=pairs_path,
            validated_data_json=None,
            output_dir=tmp_path / "analysis_with_judge",
            seed=42,
            enable_llm_judge=True,
            judge_model_name="stub-model",
        )
    finally:
        analyzer.OpenAIJudgeClient = original_client  # type: ignore[assignment]

    assert output_paths["llm_judge"].exists()
    assert output_paths["llm_judge_summary"].exists()
    assert output_paths["review_queue"].exists()
    assert output_paths["review_queue_summary"].exists()

    queue_rows = [
        json.loads(line)
        for line in output_paths["review_queue"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(queue_rows) == 1
    assert queue_rows[0]["trace_id"] == "pair-run-analysis-judge"
    assert queue_rows[0]["queue_reason"] in {"recommendation_only", "human_review_flag"}
    assert queue_rows[0]["judge_recommendations"] == ["Minor wording cleanup"]

    queue_summary = json.loads(output_paths["review_queue_summary"].read_text(encoding="utf-8"))
    assert queue_summary["row_count"] == 1
    assert queue_summary["summary"]["row_count"] == 1
    assert "reason_counts" in queue_summary["summary"]
    assert "top_recommendation_starts" in queue_summary["summary"]
