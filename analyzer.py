"""Phase 5 failure analysis and labeling for resume-job pairs.

This module computes rubric-defined semantic failure metrics for each valid
ResumeJobPair and writes analysis artifacts used for downstream adjudication and
visualization.

Policy note:
- Runtime policy values are intentionally defined in code for portfolio simplicity.
- Keep these synchronized with config/benchmark_policy_v1.yaml.
- This module does NOT load YAML at runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Protocol

from dotenv import load_dotenv
# Load API credentials from shared mini-projects/.env.local
load_dotenv(dotenv_path="../.env.local")

from llm_retry import call_with_backoff, maybe_batch_delay
from pydantic import BaseModel, Field, ValidationError

from schemas.models import FitLevel, ProficiencyLevel, ResumeJobPair


# NOTE: Keep synchronized with config/benchmark_policy_v1.yaml -> fit_band_policy.
FIT_BANDS: dict[FitLevel, tuple[float, float]] = {
    FitLevel.EXCELLENT: (0.80, 1.00),
    FitLevel.GOOD: (0.60, 0.79),
    FitLevel.PARTIAL: (0.40, 0.59),
    FitLevel.POOR: (0.20, 0.39),
    FitLevel.COMPLETE_MISMATCH: (0.00, 0.19),
}

# NOTE: Keep synchronized with config/benchmark_policy_v1.yaml -> metrics.missing_core_skills.
CORE_SKILL_COUNT = 3

# NOTE: Keep synchronized with config/benchmark_policy_v1.yaml -> metrics.hallucination_heuristics.
EXPERT_SKILL_MAX_FOR_ENTRY_LEVEL = 10
TOTAL_SKILL_MAX_FOR_ENTRY_LEVEL = 20
SUSPICIOUS_PHRASES: tuple[str, ...] = (
    "expert in all",
    "certified in everything",
)

# NOTE: Keep synchronized with config/benchmark_policy_v1.yaml -> metrics.awkward_language_heuristics.
BUZZWORD_THRESHOLD = 5
REPEATED_TOKEN_WINDOW = 10
REPEATED_TOKEN_COUNT = 3
LOCAL_REPETITION_WINDOW = 8
LOCAL_REPETITION_COUNT = 3
LOCAL_REPETITION_MIN_TOKEN_LENGTH = 4

# Curated starter list (rubric recommendation: 30+).
BUZZWORDS: tuple[str, ...] = (
    "synergy",
    "leverage",
    "leveraging",
    "dynamic",
    "results-driven",
    "passionate",
    "innovative",
    "cutting-edge",
    "best-in-class",
    "visionary",
    "strategic",
    "thought leader",
    "proactive",
    "self-starter",
    "team player",
    "detail-oriented",
    "go-getter",
    "fast-paced",
    "game-changer",
    "robust",
    "scalable",
    "paradigm",
    "value-add",
    "impactful",
    "mission-critical",
    "cross-functional",
    "transformational",
    "stakeholder management",
    "outside the box",
    "move the needle",
    "world-class",
    "disruptive",
    "seamless",
)

# NOTE: Keep synchronized with config/benchmark_policy_v1.yaml -> adjudication.
POLICY_ID = "benchmark_policy_v1"
OVERLAP_MARGIN_FROM_BOUNDARY_MAX = 0.03
RANDOM_AUDIT_RATE = 0.10
DEFAULT_JUDGE_MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
RECOMMENDATION_REVIEW_SCORE_MAX = 0.70

ADJUDICATION_OUTPUT_FIELDS: tuple[str, ...] = (
    "trace_id",
    "generated_fit_label",
    "computed_fit_label",
    "adjudicated_fit_label",
    "adjudicator_type",
    "reason_codes",
    "notes",
    "policy_id",
    "timestamp_utc",
)

SENIORITY_TO_SCORE: dict[str, int] = {
    "entry_level": 0,
    "mid_level": 1,
    "senior": 2,
    "lead": 3,
    "executive": 4,
}


@dataclass(frozen=True)
class PairAnalysis:
    """Computed analysis for one resume-job pair."""

    trace_id: str
    generated_fit_label: str | None
    computed_fit_label: str
    skills_overlap: float
    experience_mismatch: bool
    seniority_mismatch: bool
    missing_core_skills: bool
    hallucinated_skills: bool
    awkward_language: bool
    failure_count: int
    resume_template: str
    requires_human_review: bool
    review_reason_codes: list[str]


class JudgeEvaluation(BaseModel):
    """Typed schema for optional LLM-as-Judge outputs."""

    has_hallucinations: bool
    hallucination_details: str
    has_awkward_language: bool
    awkward_language_details: str
    overall_quality_score: float = Field(ge=0.0, le=1.0)
    fit_assessment: str
    recommendations: list[str]
    red_flags: list[str]


class JudgeClient(Protocol):
    """Protocol for LLM judge adapters used by analyzer."""

    def evaluate_pair(self, pair: ResumeJobPair, rule_analysis: PairAnalysis) -> Any:
        """Return a judge payload for one resume-job pair."""


class OpenAIJudgeClient:
    """OpenAI-compatible implementation for optional LLM-as-Judge evaluation."""

    def __init__(self, model_name: str = DEFAULT_JUDGE_MODEL_NAME) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found. Set it in mini-projects/.env.local to use --enable-llm-judge."
            )

        openai_kwargs: dict[str, Any] = {"api_key": api_key}
        base_url = os.getenv("OPENAI_BASE_URL")
        if base_url:
            openai_kwargs["base_url"] = base_url

        from openai import OpenAI  # Local import keeps analyzer usable without network deps at import time.

        self._client = OpenAI(**openai_kwargs)
        self._model_name = model_name

    def evaluate_pair(self, pair: ResumeJobPair, rule_analysis: PairAnalysis) -> Any:
        """Evaluate pair quality with an LLM judge and return parsed JSON object."""
        prompt = build_judge_prompt(pair=pair, rule_analysis=rule_analysis)
        response = call_with_backoff(
            lambda: self._client.chat.completions.create(
                model=self._model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict resume quality judge. "
                            "Return ONLY one valid JSON object with the required fields."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=1200,
            )
        )
        maybe_batch_delay()
        content = response.choices[0].message.content or ""
        return extract_json_object(content)


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract JSON object from model output (with fenced-code tolerance)."""
    stripped = text.strip()

    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    first_brace = stripped.find("{")
    last_brace = stripped.rfind("}")
    if first_brace == -1 or last_brace == -1 or last_brace <= first_brace:
        raise ValueError("LLM judge output did not include a JSON object.")

    payload = json.loads(stripped[first_brace : last_brace + 1])
    if not isinstance(payload, dict):
        raise ValueError("LLM judge output must be a JSON object.")
    return payload


def build_judge_prompt(pair: ResumeJobPair, rule_analysis: PairAnalysis) -> str:
    """Build holistic judge prompt with pair context + rule-metric context."""
    resume_payload = pair.resume.model_dump(mode="json")
    job_payload = pair.job_description.model_dump(mode="json")

    rule_context = {
        "computed_fit_label": rule_analysis.computed_fit_label,
        "skills_overlap": rule_analysis.skills_overlap,
        "experience_mismatch": rule_analysis.experience_mismatch,
        "seniority_mismatch": rule_analysis.seniority_mismatch,
        "missing_core_skills": rule_analysis.missing_core_skills,
        "hallucinated_skills": rule_analysis.hallucinated_skills,
        "awkward_language": rule_analysis.awkward_language,
    }

    return (
        "Evaluate this resume-job pair using the rubric below. "
        "Use contextual judgment and evidence from the provided job + resume.\n\n"
        "Evaluation rubric:\n"
        "1) Hallucinations\n"
        "- Definition: unverifiable or implausible claims, inflated seniority relative to experience, "
        "or timeline inconsistencies.\n"
        "- Positive examples (flag):\n"
        "  - Candidate claims lead/principal scope with minimal total experience and no supporting evidence.\n"
        "  - Resume lists many 'expert' skills across unrelated stacks without concrete project outcomes.\n"
        "  - Dates or role progression are internally inconsistent.\n"
        "- Negative examples (do not auto-flag):\n"
        "  - Strong claims supported by specific, plausible outcomes and scope.\n"
        "  - Career switch with realistic progression and coherent timeline.\n\n"
        "2) Awkward Language\n"
        "- Definition: excessive jargon, unnatural phrasing, repetitive AI-like wording, or buzzword stuffing "
        "that reduces clarity.\n"
        "- Positive examples (flag):\n"
        "  - Dense buzzword chains with little concrete content.\n"
        "  - Repetitive generic claims not grounded in accomplishments.\n"
        "  - Local repetition patterns (same word 3+ times in close proximity), e.g., 'synergy synergy synergy'.\n"
        "- Negative examples (do not auto-flag):\n"
        "  - Concise professional language with standard technical terminology.\n"
        "  - Minor stylistic quirks that do not harm readability.\n\n"
        "3) Fit Assessment\n"
        "- Definition: holistic alignment of required skills, relevant experience, and seniority with the job.\n"
        "- Positive examples (good fit):\n"
        "  - Required skills and experience are directly evidenced in roles/achievements.\n"
        "- Negative examples (poor fit):\n"
        "  - Missing multiple required skills or major seniority/experience mismatch.\n\n"
        "4) Red Flags\n"
        "- Definition: concerns that warrant review, such as unexplained employment gaps or inconsistent progression.\n"
        "- Positive examples (include in red_flags):\n"
        "  - Unexplained long gaps, conflicting dates, abrupt title jumps without evidence.\n"
        "- Negative examples (do not include):\n"
        "  - Gaps with clear explanation, or non-linear but coherent career path.\n\n"
        "Output requirements:\n"
        "- Return ONLY one JSON object with EXACT keys:\n"
        "{\n"
        "  \"has_hallucinations\": boolean,\n"
        "  \"hallucination_details\": string,\n"
        "  \"has_awkward_language\": boolean,\n"
        "  \"awkward_language_details\": string,\n"
        "  \"overall_quality_score\": number between 0.0 and 1.0,\n"
        "  \"fit_assessment\": string,\n"
        "  \"recommendations\": array of strings,\n"
        "  \"red_flags\": array of strings\n"
        "}\n"
        "- For hallucination_details and awkward_language_details, cite 1-3 concrete snippets or facts from the "
        "resume/job that justify the decision.\n"
        "- If uncertain, be conservative: avoid hard flags and use recommendations for follow-up checks.\n\n"
        "Rules-based context (for reference, may be incomplete):\n"
        f"{json.dumps(rule_context, ensure_ascii=True)}\n\n"
        "Job Description:\n"
        f"{json.dumps(job_payload, ensure_ascii=True)}\n\n"
        "Resume:\n"
        f"{json.dumps(resume_payload, ensure_ascii=True)}\n"
    )


def normalize_skill(skill: str) -> str:
    """Normalize skill strings for overlap comparison."""
    normalized = skill.lower().strip()
    normalized = re.sub(r"\b(developer|engineer)\b", "", normalized)
    normalized = re.sub(r"\.js\b", "", normalized)
    normalized = re.sub(r"\d+(?:\.\d+)?", "", normalized)
    normalized = re.sub(r"[^a-z0-9+\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def jaccard_overlap(resume_skills: set[str], required_skills: set[str]) -> float:
    """Compute Jaccard overlap between normalized resume and required skill sets."""
    if not resume_skills and not required_skills:
        return 1.0
    if not resume_skills or not required_skills:
        return 0.0

    intersection = resume_skills.intersection(required_skills)
    union = resume_skills.union(required_skills)
    return len(intersection) / len(union)


def years_between(start: date, end: date) -> float:
    """Compute fractional years between two dates."""
    return max(0.0, (end - start).days / 365.25)


def estimate_resume_years(pair: ResumeJobPair, reference_date: date | None = None) -> float:
    """Estimate total years of experience from resume experience timeline."""
    end_ref = reference_date or datetime.now(timezone.utc).date()
    total = 0.0
    for exp in pair.resume.experience:
        end = exp.end_date or end_ref
        total += years_between(exp.start_date, end)
    return total


def infer_resume_seniority(pair: ResumeJobPair) -> str:
    """Infer seniority level from years of experience and title keywords."""
    years = estimate_resume_years(pair)
    titles = " ".join(exp.position.lower() for exp in pair.resume.experience)

    if any(keyword in titles for keyword in ("vp", "vice president", "chief", "cto", "ceo", "executive")):
        return "executive"
    if any(keyword in titles for keyword in ("director", "head of", "lead", "principal")):
        return "lead"
    if "senior" in titles or years >= 5.0:
        return "senior"
    if years >= 2.0:
        return "mid_level"
    return "entry_level"


def map_overlap_to_fit_level(overlap: float, missing_core_skills: bool) -> FitLevel:
    """Map overlap score into configured fit bands."""
    if overlap >= FIT_BANDS[FitLevel.EXCELLENT][0]:
        if missing_core_skills:
            return FitLevel.GOOD
        return FitLevel.EXCELLENT
    if overlap >= FIT_BANDS[FitLevel.GOOD][0]:
        return FitLevel.GOOD
    if overlap >= FIT_BANDS[FitLevel.PARTIAL][0]:
        return FitLevel.PARTIAL
    if overlap >= FIT_BANDS[FitLevel.POOR][0]:
        return FitLevel.POOR
    return FitLevel.COMPLETE_MISMATCH


def count_buzzwords(text: str) -> int:
    """Count occurrences of curated buzzword phrases."""
    lowered = text.lower()
    return sum(lowered.count(term) for term in BUZZWORDS)


def has_repeated_token_pattern(text: str) -> bool:
    """Detect repeated token windows indicating awkward/generated language."""
    tokens = re.findall(r"[a-z0-9']+", text.lower())
    if len(tokens) < REPEATED_TOKEN_WINDOW:
        return False

    windows: dict[tuple[str, ...], int] = {}
    for i in range(0, len(tokens) - REPEATED_TOKEN_WINDOW + 1):
        window = tuple(tokens[i : i + REPEATED_TOKEN_WINDOW])
        windows[window] = windows.get(window, 0) + 1
        if windows[window] >= REPEATED_TOKEN_COUNT:
            return True
    return False


def has_local_token_repetition(text: str) -> bool:
    """Detect local word repetition (same token 3+ times within a short span)."""
    tokens = re.findall(r"[a-z0-9']+", text.lower())
    if len(tokens) < LOCAL_REPETITION_WINDOW:
        return False

    for i in range(0, len(tokens) - LOCAL_REPETITION_WINDOW + 1):
        window = tokens[i : i + LOCAL_REPETITION_WINDOW]
        counts: dict[str, int] = {}
        for token in window:
            if len(token) < LOCAL_REPETITION_MIN_TOKEN_LENGTH:
                continue
            counts[token] = counts.get(token, 0) + 1
            if counts[token] >= LOCAL_REPETITION_COUNT:
                return True
    return False


def build_resume_text_blob(pair: ResumeJobPair) -> str:
    """Build free-text blob for language heuristics."""
    pieces: list[str] = []
    if pair.resume.summary:
        pieces.append(pair.resume.summary)
    for exp in pair.resume.experience:
        pieces.append(exp.position)
        pieces.append(exp.description)
        pieces.extend(exp.achievements)
    pieces.extend(skill.name for skill in pair.resume.skills)
    return "\n".join(pieces)


def near_fit_boundary(overlap: float) -> bool:
    """True when overlap is close to any fit band boundary."""
    boundaries = [
        FIT_BANDS[FitLevel.EXCELLENT][0],
        FIT_BANDS[FitLevel.GOOD][0],
        FIT_BANDS[FitLevel.PARTIAL][0],
        FIT_BANDS[FitLevel.POOR][0],
    ]
    return any(abs(overlap - boundary) <= OVERLAP_MARGIN_FROM_BOUNDARY_MAX for boundary in boundaries)


def deterministic_random_audit(trace_id: str) -> bool:
    """Deterministic random-audit selection based on trace_id hash."""
    digest = hashlib.sha256(trace_id.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    return value < RANDOM_AUDIT_RATE


def analyze_pair(pair: ResumeJobPair) -> PairAnalysis:
    """Compute all Phase 5 metrics and adjudication signals for one pair."""
    normalized_resume_skills = {
        normalize_skill(skill.name) for skill in pair.resume.skills if normalize_skill(skill.name)
    }
    normalized_required_skills = {
        normalize_skill(skill) for skill in pair.job_description.requirements.required_skills if normalize_skill(skill)
    }

    overlap = jaccard_overlap(normalized_resume_skills, normalized_required_skills)

    required_years = pair.job_description.requirements.years_experience
    resume_years = estimate_resume_years(pair)
    experience_mismatch = bool(required_years and resume_years < (0.5 * required_years))

    job_level_raw = pair.job_description.requirements.experience_level
    job_seniority = job_level_raw.value if hasattr(job_level_raw, "value") else str(job_level_raw)
    resume_seniority = infer_resume_seniority(pair)
    seniority_mismatch = abs(SENIORITY_TO_SCORE[resume_seniority] - SENIORITY_TO_SCORE[job_seniority]) > 1

    core_required = [
        normalize_skill(skill)
        for skill in pair.job_description.requirements.required_skills[:CORE_SKILL_COUNT]
        if normalize_skill(skill)
    ]
    missing_core_skills = any(skill not in normalized_resume_skills for skill in core_required)

    expert_skill_count = sum(1 for skill in pair.resume.skills if skill.proficiency == ProficiencyLevel.EXPERT)
    total_skill_count = len(pair.resume.skills)
    suspicious_text = build_resume_text_blob(pair).lower()

    hallucinated_skills = False
    if resume_seniority == "entry_level":
        hallucinated_skills = (
            expert_skill_count > EXPERT_SKILL_MAX_FOR_ENTRY_LEVEL
            or total_skill_count > TOTAL_SKILL_MAX_FOR_ENTRY_LEVEL
        )

    if any(phrase in suspicious_text for phrase in SUSPICIOUS_PHRASES):
        hallucinated_skills = True

    if resume_seniority == "entry_level" and expert_skill_count >= 15:
        hallucinated_skills = True

    buzzword_count = count_buzzwords(suspicious_text)
    awkward_language = (
        buzzword_count > BUZZWORD_THRESHOLD
        or has_repeated_token_pattern(suspicious_text)
        or has_local_token_repetition(suspicious_text)
    )

    computed_fit = map_overlap_to_fit_level(overlap, missing_core_skills)
    generated_fit = pair.fit_level.value if pair.fit_level else None

    failure_flags = [
        experience_mismatch,
        seniority_mismatch,
        missing_core_skills,
        hallucinated_skills,
        awkward_language,
    ]
    failure_count = sum(1 for flag in failure_flags if flag)

    reason_codes: list[str] = []
    if generated_fit and generated_fit != computed_fit.value:
        reason_codes.append("fit_label_mismatch")
    if computed_fit == FitLevel.EXCELLENT and hallucinated_skills:
        reason_codes.append("excellent_with_hallucination")
    if near_fit_boundary(overlap):
        reason_codes.append("near_fit_boundary")

    inconsistent_profile = generated_fit == "excellent" and failure_count >= 2
    if inconsistent_profile:
        reason_codes.append("rule_conflict_profile")

    ambiguous_language = awkward_language and buzzword_count <= BUZZWORD_THRESHOLD + 2
    if ambiguous_language:
        reason_codes.append("ambiguous_language")

    if deterministic_random_audit(pair.trace_id):
        reason_codes.append("random_audit")

    requires_human_review = bool(reason_codes)

    return PairAnalysis(
        trace_id=pair.trace_id,
        generated_fit_label=generated_fit,
        computed_fit_label=computed_fit.value,
        skills_overlap=round(overlap, 6),
        experience_mismatch=experience_mismatch,
        seniority_mismatch=seniority_mismatch,
        missing_core_skills=missing_core_skills,
        hallucinated_skills=hallucinated_skills,
        awkward_language=awkward_language,
        failure_count=failure_count,
        resume_template=pair.resume.metadata.prompt_template,
        requires_human_review=requires_human_review,
        review_reason_codes=reason_codes,
    )


def parse_pairs_from_jsonl(file_path: Path) -> list[ResumeJobPair]:
    """Read and validate pairs from a JSONL file."""
    pairs: list[ResumeJobPair] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            payload = json.loads(raw)
            try:
                pair = ResumeJobPair.model_validate(payload)
            except Exception as exc:
                raise ValueError(
                    f"Invalid pair record in {file_path} at line {line_number}: {exc}"
                ) from exc
            pairs.append(pair)
    return pairs


def parse_pairs_from_validated_data(file_path: Path) -> list[ResumeJobPair]:
    """Read pair records from Phase 4 validated_data JSON artifact."""
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    records = payload.get("records", {})
    pair_payloads = records.get("pairs", [])
    return [ResumeJobPair.model_validate(item) for item in pair_payloads]


def find_latest_file(directory: Path, pattern: str) -> Path | None:
    """Return newest file matching pattern in directory."""
    matches = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not matches:
        return None
    return matches[-1]


def pearson(values_a: list[float], values_b: list[float]) -> float:
    """Compute Pearson correlation coefficient for equal-length series."""
    if len(values_a) != len(values_b) or len(values_a) < 2:
        return 0.0
    mean_a = mean(values_a)
    mean_b = mean(values_b)

    numerator = sum((a - mean_a) * (b - mean_b) for a, b in zip(values_a, values_b, strict=True))
    denom_a = sum((a - mean_a) ** 2 for a in values_a) ** 0.5
    denom_b = sum((b - mean_b) ** 2 for b in values_b) ** 0.5

    if denom_a == 0.0 or denom_b == 0.0:
        return 0.0
    return numerator / (denom_a * denom_b)


def build_correlation_matrix(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Build metric correlation matrix from analysis rows."""
    keys = [
        "skills_overlap",
        "experience_mismatch",
        "seniority_mismatch",
        "missing_core_skills",
        "hallucinated_skills",
        "awkward_language",
    ]

    numeric: dict[str, list[float]] = {}
    for key in keys:
        if key == "skills_overlap":
            numeric[key] = [float(row[key]) for row in rows]
        else:
            numeric[key] = [1.0 if row[key] else 0.0 for row in rows]

    matrix: dict[str, dict[str, float]] = {}
    for key_a in keys:
        matrix[key_a] = {}
        for key_b in keys:
            matrix[key_a][key_b] = round(pearson(numeric[key_a], numeric[key_b]), 4)
    return matrix


def analyze_pairs(
    pairs: list[ResumeJobPair],
    judge_client: JudgeClient | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run analysis + adjudication, optionally enriched with LLM-as-Judge signals."""
    labels: list[dict[str, Any]] = []
    adjudication_log: list[dict[str, Any]] = []
    judge_rows: list[dict[str, Any]] = []
    total_pairs = len(pairs)

    if judge_client is not None:
        print(f"LLM judge enabled: processing {total_pairs} pair(s)...", flush=True)

    for index, pair in enumerate(pairs, start=1):
        result = analyze_pair(pair)
        requires_human_review = result.requires_human_review
        reason_codes = list(result.review_reason_codes)

        judge_evaluation: JudgeEvaluation | None = None
        judge_error: str | None = None
        if judge_client is not None:
            try:
                raw_judge = judge_client.evaluate_pair(pair, result)
                judge_evaluation = JudgeEvaluation.model_validate(raw_judge)
            except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                judge_error = str(exc)
                requires_human_review = True
                if "llm_judge_error" not in reason_codes:
                    reason_codes.append("llm_judge_error")
            except Exception as exc:  # pragma: no cover - external provider/network path
                judge_error = str(exc)
                requires_human_review = True
                if "llm_judge_error" not in reason_codes:
                    reason_codes.append("llm_judge_error")

            if judge_evaluation is not None:
                judge_flagged = (
                    judge_evaluation.has_hallucinations
                    or judge_evaluation.has_awkward_language
                    or bool(judge_evaluation.red_flags)
                    or judge_evaluation.overall_quality_score < 0.40
                )
                if judge_flagged:
                    requires_human_review = True
                    if "llm_judge_flagged" not in reason_codes:
                        reason_codes.append("llm_judge_flagged")

                recommendation_triggered_review = (
                    bool(judge_evaluation.recommendations)
                    and judge_evaluation.overall_quality_score < RECOMMENDATION_REVIEW_SCORE_MAX
                )
                if recommendation_triggered_review:
                    requires_human_review = True
                    if "llm_judge_recommendation" not in reason_codes:
                        reason_codes.append("llm_judge_recommendation")

                judge_rows.append(
                    {
                        "trace_id": pair.trace_id,
                        "has_hallucinations": judge_evaluation.has_hallucinations,
                        "hallucination_details": judge_evaluation.hallucination_details,
                        "has_awkward_language": judge_evaluation.has_awkward_language,
                        "awkward_language_details": judge_evaluation.awkward_language_details,
                        "overall_quality_score": round(judge_evaluation.overall_quality_score, 4),
                        "fit_assessment": judge_evaluation.fit_assessment,
                        "recommendations": judge_evaluation.recommendations,
                        "red_flags": judge_evaluation.red_flags,
                        "judge_error": None,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    }
                )
            else:
                judge_rows.append(
                    {
                        "trace_id": pair.trace_id,
                        "has_hallucinations": None,
                        "hallucination_details": "",
                        "has_awkward_language": None,
                        "awkward_language_details": "",
                        "overall_quality_score": None,
                        "fit_assessment": "",
                        "recommendations": [],
                        "red_flags": [],
                        "judge_error": judge_error,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    }
                )

            if index == 1 or index % 25 == 0 or index == total_pairs:
                print(
                    f"LLM judge progress: {index}/{total_pairs} pairs processed",
                    flush=True,
                )

        label_row = {
            "trace_id": result.trace_id,
            "generated_fit_label": result.generated_fit_label,
            "computed_fit_label": result.computed_fit_label,
            "skills_overlap": result.skills_overlap,
            "experience_mismatch": result.experience_mismatch,
            "seniority_mismatch": result.seniority_mismatch,
            "missing_core_skills": result.missing_core_skills,
            "hallucinated_skills": result.hallucinated_skills,
            "awkward_language": result.awkward_language,
            "failure_count": result.failure_count,
            "resume_template": result.resume_template,
            "requires_human_review": requires_human_review,
            "review_reason_codes": reason_codes,
        }
        labels.append(label_row)

        if requires_human_review:
            adjudicated_fit = result.computed_fit_label
            adjudicator_type = "human_review"
            notes = "Flagged for review by policy trigger(s)."
        else:
            adjudicated_fit = result.computed_fit_label
            adjudicator_type = "auto_accept"
            notes = "Auto-accepted by policy checks."

        adjudication_row = {
            "trace_id": result.trace_id,
            "generated_fit_label": result.generated_fit_label,
            "computed_fit_label": result.computed_fit_label,
            "adjudicated_fit_label": adjudicated_fit,
            "adjudicator_type": adjudicator_type,
            "reason_codes": reason_codes,
            "notes": notes,
            "policy_id": POLICY_ID,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        adjudication_log.append(adjudication_row)

    return labels, adjudication_log, judge_rows


def summarize_judge_rows(judge_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute summary statistics for optional LLM judge outputs."""
    if not judge_rows:
        return {
            "row_count": 0,
            "judge_error_count": 0,
            "hallucination_rate": 0.0,
            "awkward_language_rate": 0.0,
            "mean_overall_quality_score": 0.0,
            "red_flag_rate": 0.0,
        }

    valid_rows = [row for row in judge_rows if row.get("judge_error") is None]
    judge_error_count = len(judge_rows) - len(valid_rows)
    if not valid_rows:
        return {
            "row_count": len(judge_rows),
            "judge_error_count": judge_error_count,
            "hallucination_rate": 0.0,
            "awkward_language_rate": 0.0,
            "mean_overall_quality_score": 0.0,
            "red_flag_rate": 0.0,
        }

    return {
        "row_count": len(judge_rows),
        "judge_error_count": judge_error_count,
        "hallucination_rate": round(
            mean(1.0 if bool(row.get("has_hallucinations")) else 0.0 for row in valid_rows), 4
        ),
        "awkward_language_rate": round(
            mean(1.0 if bool(row.get("has_awkward_language")) else 0.0 for row in valid_rows), 4
        ),
        "mean_overall_quality_score": round(
            mean(float(row["overall_quality_score"]) for row in valid_rows), 4
        ),
        "red_flag_rate": round(
            mean(1.0 if bool(row.get("red_flags")) else 0.0 for row in valid_rows), 4
        ),
    }


def build_review_queue(
    labels: list[dict[str, Any]],
    adjudication_log: list[dict[str, Any]],
    judge_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build review queue rows from hard flags plus recommendation-only judge outputs."""
    label_by_trace = {row["trace_id"]: row for row in labels}
    adjudication_by_trace = {row["trace_id"]: row for row in adjudication_log}
    judge_by_trace = {str(row.get("trace_id")): row for row in judge_rows}

    queue_rows: list[dict[str, Any]] = []
    queued_trace_ids: set[str] = set()

    for trace_id, label_row in label_by_trace.items():
        if not bool(label_row.get("requires_human_review")):
            continue

        adjudication_row = adjudication_by_trace.get(trace_id, {})
        judge_row = judge_by_trace.get(trace_id, {})
        queue_rows.append(
            {
                "trace_id": trace_id,
                "queue_reason": "human_review_flag",
                "requires_human_review": True,
                "review_reason_codes": label_row.get("review_reason_codes", []),
                "computed_fit_label": label_row.get("computed_fit_label"),
                "adjudicated_fit_label": adjudication_row.get("adjudicated_fit_label"),
                "judge_overall_quality_score": judge_row.get("overall_quality_score"),
                "judge_recommendations": judge_row.get("recommendations", []),
                "judge_red_flags": judge_row.get("red_flags", []),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        queued_trace_ids.add(trace_id)

    for judge_row in judge_rows:
        trace_id = str(judge_row.get("trace_id"))
        if trace_id in queued_trace_ids:
            continue
        if judge_row.get("judge_error") is not None:
            continue

        recommendations = judge_row.get("recommendations")
        if not isinstance(recommendations, list) or not recommendations:
            continue

        label_row = label_by_trace.get(trace_id)
        adjudication_row = adjudication_by_trace.get(trace_id, {})
        queue_rows.append(
            {
                "trace_id": trace_id,
                "queue_reason": "recommendation_only",
                "requires_human_review": bool(label_row and label_row.get("requires_human_review")),
                "review_reason_codes": (label_row or {}).get("review_reason_codes", []),
                "computed_fit_label": (label_row or {}).get("computed_fit_label"),
                "adjudicated_fit_label": adjudication_row.get("adjudicated_fit_label"),
                "judge_overall_quality_score": judge_row.get("overall_quality_score"),
                "judge_recommendations": recommendations,
                "judge_red_flags": judge_row.get("red_flags", []),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
        )

    return queue_rows


def summarize_review_queue(queue_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute compact summary stats for review queue routing."""
    if not queue_rows:
        return {
            "row_count": 0,
            "requires_human_review_count": 0,
            "recommendation_only_count": 0,
            "reason_counts": {},
            "top_recommendation_starts": [],
        }

    reason_counts: dict[str, int] = {}
    recommendation_starts: dict[str, int] = {}
    requires_human_review_count = 0

    for row in queue_rows:
        queue_reason = str(row.get("queue_reason", "unknown"))
        reason_counts[queue_reason] = reason_counts.get(queue_reason, 0) + 1

        if bool(row.get("requires_human_review")):
            requires_human_review_count += 1

        for recommendation in row.get("judge_recommendations", []):
            if not isinstance(recommendation, str) or not recommendation.strip():
                continue
            start = recommendation.strip().split(".")[0][:80]
            recommendation_starts[start] = recommendation_starts.get(start, 0) + 1

    top_recommendation_starts = sorted(
        recommendation_starts.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:5]

    return {
        "row_count": len(queue_rows),
        "requires_human_review_count": requires_human_review_count,
        "recommendation_only_count": reason_counts.get("recommendation_only", 0),
        "reason_counts": reason_counts,
        "top_recommendation_starts": [
            {"text": text, "count": count} for text, count in top_recommendation_starts
        ],
    }


def breakdown_by_fit_level(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Compute per-fit-level failure rates."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = row["generated_fit_label"] or "unlabeled"
        grouped.setdefault(key, []).append(row)

    metrics = [
        "experience_mismatch",
        "seniority_mismatch",
        "missing_core_skills",
        "hallucinated_skills",
        "awkward_language",
    ]

    output: dict[str, dict[str, float]] = {}
    for fit_level, values in grouped.items():
        output[fit_level] = {
            "count": float(len(values)),
            "mean_overlap": round(mean(float(v["skills_overlap"]) for v in values), 4),
            "mean_failure_count": round(mean(float(v["failure_count"]) for v in values), 4),
        }
        for metric in metrics:
            output[fit_level][f"{metric}_rate"] = round(
                mean(1.0 if v[metric] else 0.0 for v in values), 4
            )
    return output


def breakdown_by_template(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Compute per-template failure rates."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["resume_template"], []).append(row)

    output: dict[str, dict[str, float]] = {}
    for template, values in grouped.items():
        output[template] = {
            "count": float(len(values)),
            "mean_overlap": round(mean(float(v["skills_overlap"]) for v in values), 4),
            "mean_failure_count": round(mean(float(v["failure_count"]) for v in values), 4),
            "awkward_language_rate": round(
                mean(1.0 if v["awkward_language"] else 0.0 for v in values), 4
            ),
            "hallucinated_skills_rate": round(
                mean(1.0 if v["hallucinated_skills"] else 0.0 for v in values), 4
            ),
        }
    return output


def write_jsonl(rows: list[dict[str, Any]], file_path: Path) -> None:
    """Write dict rows to JSONL."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True))
            handle.write("\n")


def run_analysis(
    input_pairs_jsonl: Path | None,
    validated_data_json: Path | None,
    output_dir: Path,
    seed: int,
    enable_llm_judge: bool = False,
    judge_model_name: str = DEFAULT_JUDGE_MODEL_NAME,
) -> dict[str, Path]:
    """Execute Phase 5 analysis and write output artifacts."""
    random.seed(seed)

    print("Starting Phase 5 analysis...", flush=True)

    if validated_data_json:
        pairs = parse_pairs_from_validated_data(validated_data_json)
        input_source = str(validated_data_json)
    elif input_pairs_jsonl:
        pairs = parse_pairs_from_jsonl(input_pairs_jsonl)
        input_source = str(input_pairs_jsonl)
    else:
        latest_validated = find_latest_file(Path("outputs/validation"), "validated_data_*.json")
        if latest_validated:
            pairs = parse_pairs_from_validated_data(latest_validated)
            input_source = str(latest_validated)
        else:
            latest_pairs = find_latest_file(Path("data/generated"), "pairs_*.jsonl")
            if not latest_pairs:
                raise FileNotFoundError("No input pair artifacts found for analysis.")
            pairs = parse_pairs_from_jsonl(latest_pairs)
            input_source = str(latest_pairs)

    judge_client: JudgeClient | None = None
    if enable_llm_judge:
        judge_client = OpenAIJudgeClient(model_name=judge_model_name)

    labels, adjudication_log, judge_rows = analyze_pairs(pairs, judge_client=judge_client)

    print(f"Analysis complete: {len(labels)} pair(s) processed.", flush=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    failure_labels_path = output_dir / f"failure_labels_{timestamp}.jsonl"
    correlation_path = output_dir / f"correlation_matrix_{timestamp}.json"
    failure_breakdown_path = output_dir / f"failure_breakdown_{timestamp}.json"
    adjudication_log_path = output_dir / f"adjudication_log_{timestamp}.jsonl"
    spot_check_path = output_dir / f"spot_check_{timestamp}.json"
    llm_judge_path = output_dir / f"llm_judge_{timestamp}.jsonl"
    llm_judge_summary_path = output_dir / f"llm_judge_summary_{timestamp}.json"
    review_queue_path = output_dir / f"review_queue_{timestamp}.jsonl"
    review_queue_summary_path = output_dir / f"review_queue_summary_{timestamp}.json"

    write_jsonl(labels, failure_labels_path)
    write_jsonl(adjudication_log, adjudication_log_path)

    correlation_payload = {
        "input_source": input_source,
        "row_count": len(labels),
        "matrix": build_correlation_matrix(labels),
    }
    correlation_path.write_text(json.dumps(correlation_payload, indent=2), encoding="utf-8")

    breakdown_payload = {
        "input_source": input_source,
        "row_count": len(labels),
        "by_fit_level": breakdown_by_fit_level(labels),
        "by_template": breakdown_by_template(labels),
    }
    failure_breakdown_path.write_text(json.dumps(breakdown_payload, indent=2), encoding="utf-8")

    sample = labels[:10]
    spot_check_path.write_text(
        json.dumps({"input_source": input_source, "sample_size": len(sample), "rows": sample}, indent=2),
        encoding="utf-8",
    )

    output_paths: dict[str, Path] = {
        "failure_labels": failure_labels_path,
        "correlation_matrix": correlation_path,
        "failure_breakdown": failure_breakdown_path,
        "adjudication_log": adjudication_log_path,
        "spot_check": spot_check_path,
    }

    if enable_llm_judge:
        review_queue_rows = build_review_queue(
            labels=labels,
            adjudication_log=adjudication_log,
            judge_rows=judge_rows,
        )
        write_jsonl(judge_rows, llm_judge_path)
        write_jsonl(review_queue_rows, review_queue_path)
        llm_judge_summary_payload = {
            "input_source": input_source,
            "row_count": len(judge_rows),
            "summary": summarize_judge_rows(judge_rows),
        }
        llm_judge_summary_path.write_text(
            json.dumps(llm_judge_summary_payload, indent=2),
            encoding="utf-8",
        )
        review_queue_summary_payload = {
            "input_source": input_source,
            "row_count": len(review_queue_rows),
            "summary": summarize_review_queue(review_queue_rows),
        }
        review_queue_summary_path.write_text(
            json.dumps(review_queue_summary_payload, indent=2),
            encoding="utf-8",
        )
        output_paths["llm_judge"] = llm_judge_path
        output_paths["llm_judge_summary"] = llm_judge_summary_path
        output_paths["review_queue"] = review_queue_path
        output_paths["review_queue_summary"] = review_queue_summary_path

    return output_paths


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Phase 5 failure analysis and labeling")
    parser.add_argument(
        "--pairs-jsonl",
        type=Path,
        default=None,
        help="Optional explicit pairs JSONL input path.",
    )
    parser.add_argument(
        "--validated-data-json",
        type=Path,
        default=None,
        help="Optional validated_data JSON from Phase 4.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/analysis"),
        help="Directory for analysis artifacts.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic seed for repeatable random-audit decisions.",
    )
    parser.add_argument(
        "--enable-llm-judge",
        action="store_true",
        help="Enable optional LLM-as-Judge outputs (slower/costlier).",
    )
    parser.add_argument(
        "--judge-model-name",
        type=str,
        default=DEFAULT_JUDGE_MODEL_NAME,
        help="OpenAI-compatible model name for optional LLM judge stage.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    paths = run_analysis(
        input_pairs_jsonl=args.pairs_jsonl,
        validated_data_json=args.validated_data_json,
        output_dir=args.output_dir,
        seed=args.seed,
        enable_llm_judge=args.enable_llm_judge,
        judge_model_name=args.judge_model_name,
    )

    print("=" * 72)
    print("PHASE 5 - FAILURE ANALYSIS & LABELING")
    print("=" * 72)
    print("Saved artifacts:")
    for key, path in paths.items():
        print(f"- {key}: {path}")


if __name__ == "__main__":
    main()
