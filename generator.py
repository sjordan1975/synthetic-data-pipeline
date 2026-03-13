"""Phase 3: Multi-template synthetic generation for resume-job matching.

This module scales generation from a single template to a stratified multi-template
pipeline while keeping fit-level control and schema safety:

1. Generate jobs across 5+ styles with explicit niche-role coverage.
2. Generate resumes across 5+ styles conditioned on fit-level constraints.
3. Build traceable ResumeJobPair records.
4. Save timestamped JSONL artifacts plus coverage summaries.
"""

from __future__ import annotations

import argparse
import os
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable
from uuid import uuid4

import instructor
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

from constraint_injector import build_fit_constraint, render_constraint_block
from llm_retry import call_with_backoff, maybe_batch_delay
from schemas.models import (
    ContactInfo,
    Education,
    Experience,
    FitLevel,
    JobDescription,
    JobMetadata,
    JobRequirements,
    ProficiencyLevel,
    Resume,
    ResumeJobPair,
    ResumeMetadata,
    SeniorityLevel,
    Skill,
)
from template_loader import TemplateLoader


# Load API credentials from shared mini-projects/.env.local
load_dotenv(dotenv_path="../.env.local")

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
JOB_REQUIREMENTS_TEMPLATE_ID = "job_requirements_v1"
TEMPLATE_LOADER = TemplateLoader()

JOB_TEMPLATE_IDS: tuple[str, ...] = (
    "job_formal_v1",
    "job_startup_v1",
    "job_technical_v1",
    "job_executive_v1",
    "job_niche_v1",
)

RESUME_TEMPLATE_IDS: tuple[str, ...] = (
    "resume_achievement_v1",
    "resume_career_changer_v1",
    "resume_entry_level_v1",
    "resume_leadership_v1",
    "resume_technical_specialist_v1",
)

STANDARD_INDUSTRY_FOCI = [
    "B2B SaaS",
    "Healthcare Technology",
    "FinTech",
    "E-commerce",
    "Cybersecurity",
]

NICHE_INDUSTRY_FOCI = [
    "Quantum Computing Systems",
    "Bioinformatics Platforms",
    "Embedded Avionics Safety",
    "Applied Cryptography Infrastructure",
    "Compiler Toolchains",
]

FIT_LEVEL_SEQUENCE: tuple[FitLevel, ...] = (
    FitLevel.EXCELLENT,
    FitLevel.GOOD,
    FitLevel.PARTIAL,
    FitLevel.POOR,
    FitLevel.COMPLETE_MISMATCH,
)

DEFAULT_OUTPUT_DIR = Path("data/generated")

# Resume generation still uses explicit fit constraints per pair.
FIT_LEVEL_INSTRUCTIONS = {
    FitLevel.EXCELLENT: "Target 80-100% overlap with required_skills; match seniority expectations.",
    FitLevel.GOOD: "Target 60-79% overlap with required_skills; mostly aligned experience.",
    FitLevel.PARTIAL: "Target 40-59% overlap with required_skills; notable but plausible gaps.",
    FitLevel.POOR: "Target 20-39% overlap with required_skills; clear skill and/or experience gaps.",
    FitLevel.COMPLETE_MISMATCH: "Target 0-19% overlap with required_skills; intentionally misaligned profile.",
}

SKILL_SUFFIXES = (" developer", " engineer", ".js")
VERSION_PATTERN = re.compile(r"\b\d+(?:\.\d+)*\b")

FIRST_NAMES = ["Jordan", "Alex", "Taylor", "Sam", "Casey", "Riley"]
LAST_NAMES = ["Kim", "Singh", "Lopez", "Nguyen", "Patel", "Morgan"]

GENERIC_UNRELATED_SKILLS = [
    "Salesforce",
    "Cold Outreach",
    "Retail Operations",
    "Customer Support",
    "Event Planning",
    "Inventory Management",
    "Payroll",
    "Canva",
    "HubSpot",
]


@dataclass(frozen=True)
class GenerationPlanItem:
    """Single assignment row for one generated resume-job pair."""

    job_index: int
    resume_index_within_job: int
    job_template_id: str
    resume_template_id: str
    fit_level: FitLevel
    niche_target: bool


def build_generation_plan(
    *,
    num_jobs: int,
    resumes_per_job: int,
    niche_ratio: float,
    seed: int,
) -> list[GenerationPlanItem]:
    """Create a deterministic, stratified plan for Phase 3 generation."""
    if num_jobs <= 0:
        raise ValueError("num_jobs must be > 0")
    if resumes_per_job <= 0:
        raise ValueError("resumes_per_job must be > 0")
    if not 0.0 <= niche_ratio <= 1.0:
        raise ValueError("niche_ratio must be between 0.0 and 1.0")

    rng = random.Random(seed)

    niche_job_count = int(round(num_jobs * niche_ratio))
    niche_flags = [True] * niche_job_count + [False] * (num_jobs - niche_job_count)
    rng.shuffle(niche_flags)

    plan: list[GenerationPlanItem] = []
    for job_index in range(num_jobs):
        for resume_index in range(resumes_per_job):
            plan.append(
                GenerationPlanItem(
                    job_index=job_index,
                    resume_index_within_job=resume_index,
                    job_template_id=JOB_TEMPLATE_IDS[job_index % len(JOB_TEMPLATE_IDS)],
                    resume_template_id=RESUME_TEMPLATE_IDS[
                        (job_index + resume_index) % len(RESUME_TEMPLATE_IDS)
                    ],
                    fit_level=FIT_LEVEL_SEQUENCE[resume_index % len(FIT_LEVEL_SEQUENCE)],
                    niche_target=niche_flags[job_index],
                )
            )

    return plan


def choose_industry_focus(rng: random.Random, niche_target: bool) -> str:
    """Pick industry focus with explicit niche coverage control."""
    source = NICHE_INDUSTRY_FOCI if niche_target else STANDARD_INDUSTRY_FOCI
    return rng.choice(source)


class GeneratedJobDetailsCore(BaseModel):
    """LLM-generated job details excluding requirements."""

    company: str = Field(..., min_length=2)
    industry: str = Field(..., min_length=2)
    company_size: str = Field(..., min_length=2)
    position: str = Field(..., min_length=2)
    location: str = Field(..., min_length=3)
    description: str = Field(..., min_length=50)
    salary_range: str | None = Field(default=None, min_length=5)
    benefits: list[str] = Field(default_factory=list)


class GeneratedResumeCore(BaseModel):
    """LLM-generated resume content before metadata/contact augmentation."""

    education: list[Education] = Field(..., min_length=1)
    experience: list[Experience] = Field(..., min_length=1)
    skills: list[Skill] = Field(..., min_length=3)


class GeneratedJobRequirementsCore(BaseModel):
    """LLM-generated job requirements payload with tolerant optional fields.

    Why tolerant?
    Some provider responses omit `experience_level` despite explicit prompting.
    We parse what is present, then coerce to strict JobRequirements.
    """

    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    experience_level: SeniorityLevel | None = None
    education_requirements: str = Field(default="Bachelor's degree or equivalent experience")
    years_experience: int | None = Field(default=None, ge=0, le=30)


def initialize_instructor_client() -> Any:
    """Initialize an Instructor-wrapped OpenAI-compatible client."""
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found. Add it to mini-projects/.env.local before running generator.py."
        )

    openai_kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        openai_kwargs["base_url"] = base_url

    return instructor.from_openai(OpenAI(**openai_kwargs))


def normalize_skill_name(raw_skill: str) -> str:
    """Normalize skill text for overlap calculations.

    Rules implemented for Phase 2:
    - lowercase
    - strip numeric versions (e.g., "python 3.11" -> "python")
    - strip common suffixes (developer, engineer, .js)
    """
    normalized = VERSION_PATTERN.sub("", raw_skill.lower()).strip()
    for suffix in SKILL_SUFFIXES:
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def jaccard_skill_overlap(resume: Resume, job: JobDescription) -> float:
    """Compute Jaccard overlap between resume skills and required job skills."""
    resume_skills = {normalize_skill_name(skill.name) for skill in resume.skills}
    required_skills = {normalize_skill_name(skill) for skill in job.requirements.required_skills}

    if not resume_skills and not required_skills:
        return 1.0
    if not resume_skills or not required_skills:
        return 0.0

    intersection = len(resume_skills & required_skills)
    union = len(resume_skills | required_skills)
    return intersection / union


def detect_niche_role(position: str, required_skills: list[str]) -> bool:
    """Simple heuristic for niche-role tagging used in metadata."""
    niche_keywords = {"quantum", "bioinformatics", "compiler", "cryptography", "embedded"}
    haystack = f"{position} {' '.join(required_skills)}".lower()
    return any(keyword in haystack for keyword in niche_keywords)


def build_job_prompt(industry_focus: str, job_template_id: str) -> str:
    """Build job-details prompt from external template."""
    template = TEMPLATE_LOADER.get_template(job_template_id)
    return template.format(industry_focus=industry_focus)


def build_job_requirements_prompt(job_details: GeneratedJobDetailsCore) -> str:
    """Build requirements prompt from external template."""
    template = TEMPLATE_LOADER.get_template(JOB_REQUIREMENTS_TEMPLATE_ID)
    return template.format(
        position=job_details.position,
        industry=job_details.industry,
        description=job_details.description,
    )


def build_resume_prompt(job: JobDescription, target_fit_level: FitLevel, resume_template_id: str) -> str:
    """Build resume prompt from external template + injected fit constraints."""
    requirements = job.requirements
    fit_instruction = FIT_LEVEL_INSTRUCTIONS[target_fit_level]
    fit_constraint = build_fit_constraint(job=job, fit_level=target_fit_level)
    template = TEMPLATE_LOADER.get_template(resume_template_id)

    return template.format(
        fit_level=target_fit_level.value,
        fit_instruction=fit_instruction,
        constraint_block=render_constraint_block(fit_constraint),
        company=job.company,
        position=job.position,
        industry=job.industry,
        required_skills=", ".join(requirements.required_skills),
        preferred_skills=", ".join(requirements.preferred_skills) if requirements.preferred_skills else "None",
        required_seniority=requirements.experience_level.value,
        required_years_experience=requirements.years_experience,
    )


def build_resume_summary(skills: list[Skill], target_fit_level: FitLevel) -> str:
    """Create a deterministic, readable summary from generated skills.

    Summary is optional in the Resume schema. We build it programmatically for
    reliability in this Phase 2 scaffold while keeping core resume sections
    generated by Instructor + Pydantic.
    """
    top_skills = ", ".join(skill.name for skill in skills[:3])
    if not top_skills:
        top_skills = "professional problem solving"
    return (
        f"Candidate targeting {target_fit_level.value} fit with demonstrated "
        f"strengths in {top_skills}."
    )


def infer_skill_category(skill_name: str) -> str:
    """Infer a broad skill category string for schema-valid skill records."""
    lower = skill_name.lower()
    if any(token in lower for token in ("python", "java", "javascript", "react", "sql", "docker", "kubernetes", "aws")):
        return "Programming"
    if any(token in lower for token in ("management", "leadership", "roadmap", "strategy")):
        return "Management"
    if any(token in lower for token in ("sales", "customer", "support", "outreach")):
        return "Business"
    return "General"


def infer_seniority_level(position: str, years_experience: int | None) -> SeniorityLevel:
    """Infer a seniority level when model output omits it."""
    title = position.lower()
    if any(token in title for token in ("senior", "lead", "principal", "staff", "director", "head", "vp")):
        return SeniorityLevel.SENIOR

    if years_experience is not None:
        if years_experience <= 1:
            return SeniorityLevel.ENTRY_LEVEL
        if years_experience >= 5:
            return SeniorityLevel.SENIOR

    return SeniorityLevel.MID_LEVEL


def coerce_job_requirements(
    generated_core: GeneratedJobRequirementsCore,
    job_details: GeneratedJobDetailsCore,
) -> JobRequirements:
    """Coerce tolerant requirement payload into strict JobRequirements."""
    required_skills = list(generated_core.required_skills)
    if not required_skills:
        required_skills = [
            "Communication",
            "Problem Solving",
            "Project Coordination",
            "Documentation",
            "Stakeholder Collaboration",
        ]

    preferred_skills = list(generated_core.preferred_skills)

    experience_level = generated_core.experience_level or infer_seniority_level(
        position=job_details.position,
        years_experience=generated_core.years_experience,
    )

    years_experience = generated_core.years_experience
    if years_experience is None:
        years_experience = {
            SeniorityLevel.ENTRY_LEVEL: 1,
            SeniorityLevel.MID_LEVEL: 3,
            SeniorityLevel.SENIOR: 6,
        }[experience_level]

    return JobRequirements(
        required_skills=required_skills,
        preferred_skills=preferred_skills,
        experience_level=experience_level,
        education_requirements=generated_core.education_requirements,
        years_experience=years_experience,
    )


def fallback_job_requirements(job_details: GeneratedJobDetailsCore) -> JobRequirements:
    """Build deterministic JobRequirements when LLM requirements generation fails."""
    default_required = [
        "Communication",
        "Problem Solving",
        "Project Coordination",
        "Documentation",
        "Stakeholder Collaboration",
    ]
    default_preferred = ["Data Analysis", "Process Improvement", "Presentation Skills"]

    inferred_level = infer_seniority_level(position=job_details.position, years_experience=None)
    years_experience = {
        SeniorityLevel.ENTRY_LEVEL: 1,
        SeniorityLevel.MID_LEVEL: 3,
        SeniorityLevel.SENIOR: 6,
    }[inferred_level]

    return JobRequirements(
        required_skills=default_required,
        preferred_skills=default_preferred,
        experience_level=inferred_level,
        education_requirements="Bachelor's degree or equivalent practical experience.",
        years_experience=years_experience,
    )


def fallback_resume_core(
    job: JobDescription,
    target_fit_level: FitLevel,
    candidate_index: int,
) -> GeneratedResumeCore:
    """Build a deterministic, schema-valid resume core when LLM generation fails.

    This keeps the Phase 2 pipeline runnable even when a provider occasionally
    returns empty/incomplete tool-call payloads for deeply nested response models.
    """
    rng = random.Random(candidate_index)
    required = list(job.requirements.required_skills)
    preferred = list(job.requirements.preferred_skills)

    overlap_ratio = {
        FitLevel.EXCELLENT: 1.0,
        FitLevel.GOOD: 0.7,
        FitLevel.PARTIAL: 0.5,
        FitLevel.POOR: 0.25,
        FitLevel.COMPLETE_MISMATCH: 0.0,
    }[target_fit_level]

    required_count = int(round(len(required) * overlap_ratio))
    selected_required = rng.sample(required, k=min(required_count, len(required))) if required_count > 0 else []

    preferred_count = 2 if target_fit_level in (FitLevel.EXCELLENT, FitLevel.GOOD) else 1
    selected_preferred = rng.sample(preferred, k=min(preferred_count, len(preferred))) if preferred else []

    if target_fit_level == FitLevel.COMPLETE_MISMATCH:
        unrelated_pool = GENERIC_UNRELATED_SKILLS
    else:
        unrelated_pool = [skill for skill in GENERIC_UNRELATED_SKILLS if skill not in selected_required]

    selected_unrelated = rng.sample(unrelated_pool, k=min(3, len(unrelated_pool)))
    skill_names = list(dict.fromkeys(selected_required + selected_preferred + selected_unrelated))
    if len(skill_names) < 3:
        skill_names.extend(["Communication", "Problem Solving", "Time Management"])
        skill_names = list(dict.fromkeys(skill_names))

    proficiency_by_fit = {
        FitLevel.EXCELLENT: ProficiencyLevel.EXPERT,
        FitLevel.GOOD: ProficiencyLevel.ADVANCED,
        FitLevel.PARTIAL: ProficiencyLevel.INTERMEDIATE,
        FitLevel.POOR: ProficiencyLevel.INTERMEDIATE,
        FitLevel.COMPLETE_MISMATCH: ProficiencyLevel.BEGINNER,
    }[target_fit_level]

    skills = [
        Skill(
            name=name,
            category=infer_skill_category(name),
            proficiency=proficiency_by_fit,
        )
        for name in skill_names[:8]
    ]

    required_years = job.requirements.years_experience or 3
    years_multiplier = {
        FitLevel.EXCELLENT: 1.2,
        FitLevel.GOOD: 1.0,
        FitLevel.PARTIAL: 0.6,
        FitLevel.POOR: 0.3,
        FitLevel.COMPLETE_MISMATCH: 0.1,
    }[target_fit_level]
    candidate_years = max(1, int(round(required_years * years_multiplier)))

    current_year = datetime.now(timezone.utc).year
    start_year = max(2005, current_year - candidate_years)

    education = [
        Education(
            institution="State University",
            degree="Bachelor of Science",
            field_of_study="Information Systems",
            graduation_year=max(2008, start_year - 1),
            gpa=3.3,
        )
    ]

    experience = [
        Experience(
            company="Northstar Solutions",
            position=job.position if target_fit_level in (FitLevel.EXCELLENT, FitLevel.GOOD) else "Associate",
            start_date=f"{start_year}-01-15",
            end_date=None,
            description=(
                f"Delivered role-relevant work aligned to {job.position} expectations, "
                "collaborating with cross-functional stakeholders and measurable outcomes."
            ),
            achievements=[
                "Improved team workflow efficiency by 18%.",
                "Contributed to process documentation and quality improvements.",
            ],
        )
    ]

    return GeneratedResumeCore(
        education=education,
        experience=experience,
        skills=skills,
    )


def build_contact_info(candidate_index: int, location: str) -> ContactInfo:
    """Create deterministic, schema-valid contact information.

    Why programmatic contact info?
    Contact fields (especially phone) have strict validators. Building these
    fields deterministically keeps Phase 2 focused on Instructor+schema
    integration for job/resume content itself.
    """
    first_name = FIRST_NAMES[candidate_index % len(FIRST_NAMES)]
    last_name = LAST_NAMES[candidate_index % len(LAST_NAMES)]
    name = f"{first_name} {last_name}"
    email = f"{first_name.lower()}.{last_name.lower()}{candidate_index}@example.com"
    phone_last_four = 1000 + (candidate_index % 9000)

    return ContactInfo(
        name=name,
        email=email,
        phone_region="US",
        phone=f"+1202555{phone_last_four:04d}",
        location=location,
    )


def generate_job_description(
    client: Any,
    industry_focus: str,
    job_template_id: str,
    niche_target: bool,
) -> JobDescription:
    """Generate one schema-valid job description with explicit template selection."""
    generated_details = call_with_backoff(
        lambda: client.chat.completions.create(
            model=MODEL_NAME,
            response_model=GeneratedJobDetailsCore,
            messages=[
                {
                    "role": "system",
                    "content": "You generate realistic, structured hiring artifacts.",
                },
                {
                    "role": "user",
                    "content": build_job_prompt(industry_focus=industry_focus, job_template_id=job_template_id),
                },
            ],
            temperature=0.7,
            max_tokens=1400,
        )
    )
    maybe_batch_delay()

    used_fallback = False

    try:
        generated_requirements_core = call_with_backoff(
            lambda: client.chat.completions.create(
                model=MODEL_NAME,
                response_model=GeneratedJobRequirementsCore,
                messages=[
                    {
                        "role": "system",
                        "content": "You generate realistic, structured hiring artifacts. Never omit required business fields.",
                    },
                    {
                        "role": "user",
                        "content": build_job_requirements_prompt(generated_details),
                    },
                ],
                temperature=0.2,
                max_tokens=1000,
                max_retries=2,
            )
        )
        maybe_batch_delay()
        generated_requirements = coerce_job_requirements(
            generated_core=generated_requirements_core,
            job_details=generated_details,
        )
    except Exception as exc:  # pragma: no cover - network/provider behavior
        print(f"Warning: job requirements fallback due to: {exc}")
        used_fallback = True
        generated_requirements = fallback_job_requirements(job_details=generated_details)

    metadata = JobMetadata(
        trace_id=f"job-{uuid4()}",
        generated_at=datetime.now(timezone.utc),
        is_niche_role=niche_target or detect_niche_role(
            generated_details.position,
            generated_requirements.required_skills,
        ),
        used_fallback=used_fallback,
    )

    return JobDescription(
        **generated_details.model_dump(),
        requirements=generated_requirements,
        metadata=metadata,
    )


def generate_resume(
    client: Any,
    job: JobDescription,
    target_fit_level: FitLevel,
    candidate_index: int,
    resume_template_id: str,
) -> Resume:
    """Generate one schema-valid resume conditioned on a job and fit target."""
    used_fallback = False

    try:
        generated_core = call_with_backoff(
            lambda: client.chat.completions.create(
                model=MODEL_NAME,
                response_model=GeneratedResumeCore,
                messages=[
                    {
                        "role": "system",
                        "content": "You generate realistic, structured candidate resumes. Never return empty arguments.",
                    },
                    {
                        "role": "user",
                        "content": build_resume_prompt(
                            job=job,
                            target_fit_level=target_fit_level,
                            resume_template_id=resume_template_id,
                        ),
                    },
                ],
                temperature=0.6,
                max_tokens=1800,
                max_retries=2,
            )
        )
        maybe_batch_delay()
    except Exception as exc:  # pragma: no cover - network/provider behavior
        print(
            f"Warning: resume generation fallback for candidate {candidate_index} "
            f"({target_fit_level.value}) due to: {exc}"
        )
        used_fallback = True
        generated_core = fallback_resume_core(
            job=job,
            target_fit_level=target_fit_level,
            candidate_index=candidate_index,
        )

    metadata = ResumeMetadata(
        trace_id=f"resume-{uuid4()}",
        generated_at=datetime.now(timezone.utc),
        prompt_template=resume_template_id,
        fit_level=target_fit_level,
        writing_style="professional",
        used_fallback=used_fallback,
    )

    return Resume(
        contact_info=build_contact_info(candidate_index=candidate_index, location=job.location),
        education=generated_core.education,
        experience=generated_core.experience,
        skills=generated_core.skills,
        metadata=metadata,
        summary=build_resume_summary(
            skills=generated_core.skills,
            target_fit_level=target_fit_level,
        ),
    )


def create_resume_job_pair(
    resume: Resume,
    job_description: JobDescription,
    fit_level: FitLevel,
) -> ResumeJobPair:
    """Create a schema-valid ResumeJobPair with generated artifacts."""
    return ResumeJobPair(
        resume=resume,
        job_description=job_description,
        fit_level=fit_level,
    )


def save_jsonl(records: Iterable[BaseModel], output_file: Path) -> None:
    """Write model records to JSONL using Pydantic JSON serialization."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json())
            handle.write("\n")


def print_generation_summary(
    jobs: list[JobDescription],
    resumes: list[Resume],
    pairs: list[ResumeJobPair],
) -> None:
    """Print fallback provenance and overlap distribution summaries."""
    job_fallback_count = sum(1 for job in jobs if job.metadata.used_fallback)
    resume_fallback_count = sum(1 for resume in resumes if resume.metadata.used_fallback)

    print("\nGeneration quality summary:")
    if jobs:
        print(
            f"- Job fallback rate: {job_fallback_count}/{len(jobs)} "
            f"({job_fallback_count / len(jobs):.0%})"
        )
    if resumes:
        print(
            f"- Resume fallback rate: {resume_fallback_count}/{len(resumes)} "
            f"({resume_fallback_count / len(resumes):.0%})"
        )

    overlaps_by_fit: dict[FitLevel, list[float]] = {}
    for pair in pairs:
        if pair.fit_level is None:
            continue
        overlap = jaccard_skill_overlap(resume=pair.resume, job=pair.job_description)
        overlaps_by_fit.setdefault(pair.fit_level, []).append(overlap)

    if overlaps_by_fit:
        print("- Overlap by fit level (mean/min/max):")
        for fit_level in FitLevel:
            values = overlaps_by_fit.get(fit_level)
            if not values:
                continue
            print(
                f"  - {fit_level.value:<17} "
                f"{mean(values):.2f}/{min(values):.2f}/{max(values):.2f}"
            )

    niche_jobs = sum(1 for job in jobs if job.metadata.is_niche_role)
    if jobs:
        print(
            f"- Niche role coverage: {niche_jobs}/{len(jobs)} "
            f"({niche_jobs / len(jobs):.0%})"
        )


def print_plan_summary(plan: list[GenerationPlanItem]) -> None:
    """Print planned distribution across templates/fit-levels/niche targets."""
    if not plan:
        print("No generation plan entries.")
        return

    job_template_counts: dict[str, int] = {}
    resume_template_counts: dict[str, int] = {}
    fit_counts: dict[str, int] = {}
    niche_count = 0

    for item in plan:
        job_template_counts[item.job_template_id] = job_template_counts.get(item.job_template_id, 0) + 1
        resume_template_counts[item.resume_template_id] = resume_template_counts.get(item.resume_template_id, 0) + 1
        fit_counts[item.fit_level.value] = fit_counts.get(item.fit_level.value, 0) + 1
        if item.niche_target:
            niche_count += 1

    print("Planned distribution:")
    print(f"- total pairs: {len(plan)}")
    print(f"- niche-targeted pairs: {niche_count} ({niche_count / len(plan):.0%})")
    print(f"- job templates: {job_template_counts}")
    print(f"- resume templates: {resume_template_counts}")
    print(f"- fit levels: {fit_counts}")


def generate_phase3_dataset(
    num_jobs: int = 50,
    resumes_per_job: int = 5,
    niche_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[JobDescription], list[Resume], list[ResumeJobPair]]:
    """Generate Phase 3 multi-template dataset (default 50 jobs x 5 resumes = 250 pairs)."""
    rng = random.Random(seed)
    client = initialize_instructor_client()

    plan = build_generation_plan(
        num_jobs=num_jobs,
        resumes_per_job=resumes_per_job,
        niche_ratio=niche_ratio,
        seed=seed,
    )
    print_plan_summary(plan)

    jobs: list[JobDescription] = []
    resumes: list[Resume] = []
    pairs: list[ResumeJobPair] = []

    job_cache: dict[int, JobDescription] = {}
    candidate_index = 0

    for item in plan:
        job = job_cache.get(item.job_index)
        if job is None:
            industry_focus = choose_industry_focus(rng=rng, niche_target=item.niche_target)
            job = generate_job_description(
                client=client,
                industry_focus=industry_focus,
                job_template_id=item.job_template_id,
                niche_target=item.niche_target,
            )
            job_cache[item.job_index] = job
            jobs.append(job)

            print(
                f"Job {item.job_index + 1}/{num_jobs} generated | "
                f"template={item.job_template_id} | "
                f"niche={job.metadata.is_niche_role} | "
                f"role={job.position}"
            )

        candidate_index += 1
        resume = generate_resume(
            client=client,
            job=job,
            target_fit_level=item.fit_level,
            candidate_index=candidate_index,
            resume_template_id=item.resume_template_id,
        )
        resumes.append(resume)

        pair = create_resume_job_pair(
            resume=resume,
            job_description=job,
            fit_level=item.fit_level,
        )
        pairs.append(pair)

        overlap = jaccard_skill_overlap(resume=resume, job=job)
        print(
            f"Pair {len(pairs)}/{len(plan)} | "
            f"Job={item.job_index + 1:02d} | "
            f"ResumeTemplate={item.resume_template_id} | "
            f"Fit={item.fit_level.value:<17} | "
            f"Overlap={overlap:.2f}"
        )

    return jobs, resumes, pairs


def parse_args() -> argparse.Namespace:
    """Parse CLI args for Phase 3 generation runs."""
    parser = argparse.ArgumentParser(description="Phase 3 multi-template generator")
    parser.add_argument("--num-jobs", type=int, default=50, help="Number of jobs to generate.")
    parser.add_argument(
        "--resumes-per-job",
        type=int,
        default=5,
        help="Number of resumes generated per job.",
    )
    parser.add_argument(
        "--niche-ratio",
        type=float,
        default=0.2,
        help="Fraction of jobs that should be explicitly niche-targeted (0.0-1.0).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write generated JSONL artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    """Run Phase 3 multi-template generation pipeline."""
    args = parse_args()

    print("=" * 72)
    print("PHASE 3 - MULTI-TEMPLATE GENERATION (SCALE + DIVERSITY)")
    print("=" * 72)

    jobs, resumes, pairs = generate_phase3_dataset(
        num_jobs=args.num_jobs,
        resumes_per_job=args.resumes_per_job,
        niche_ratio=args.niche_ratio,
        seed=args.seed,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir

    jobs_path = output_dir / f"jobs_{timestamp}.jsonl"
    resumes_path = output_dir / f"resumes_{timestamp}.jsonl"
    pairs_path = output_dir / f"pairs_{timestamp}.jsonl"

    save_jsonl(jobs, jobs_path)
    save_jsonl(resumes, resumes_path)
    save_jsonl(pairs, pairs_path)

    print("\nSaved artifacts:")
    print(f"- {jobs_path}")
    print(f"- {resumes_path}")
    print(f"- {pairs_path}")
    print(f"\nSummary: jobs={len(jobs)}, resumes={len(resumes)}, pairs={len(pairs)}")
    print_generation_summary(jobs=jobs, resumes=resumes, pairs=pairs)
    print("Done.")


if __name__ == "__main__":
    main()
