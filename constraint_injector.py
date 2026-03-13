"""Constraint injection helpers for fit-conditioned generation.

This module translates fit-level policy into concrete prompt constraints.
The generator can inject these constraints into resume prompts so fit behavior
is controlled and measurable, instead of relying only on stylistic prompting.
"""

from __future__ import annotations

from dataclasses import dataclass

from schemas.models import FitLevel, JobDescription


@dataclass(frozen=True)
class FitConstraint:
    """Concrete generation constraints for a target fit band."""

    target_overlap_min: float
    target_overlap_max: float
    required_skill_count_target: int
    max_unrelated_skills: int


# NOTE: Runtime fit-band thresholds are currently defined in code for simplicity.
# Keep these synchronized with config/benchmark_policy_v1.yaml (fit_band_policy).
FIT_BANDS: dict[FitLevel, tuple[float, float]] = {
    FitLevel.EXCELLENT: (0.80, 1.00),
    FitLevel.GOOD: (0.60, 0.79),
    FitLevel.PARTIAL: (0.40, 0.59),
    FitLevel.POOR: (0.20, 0.39),
    FitLevel.COMPLETE_MISMATCH: (0.00, 0.19),
}


def build_fit_constraint(job: JobDescription, fit_level: FitLevel) -> FitConstraint:
    """Build concrete skill-level constraints from job requirements + fit target."""
    overlap_min, overlap_max = FIT_BANDS[fit_level]
    required_count = max(1, len(job.requirements.required_skills))

    if fit_level == FitLevel.EXCELLENT:
        required_target = required_count
        max_unrelated = 1
    elif fit_level == FitLevel.GOOD:
        required_target = max(1, round(required_count * 0.7))
        max_unrelated = 2
    elif fit_level == FitLevel.PARTIAL:
        required_target = max(1, round(required_count * 0.5))
        max_unrelated = 3
    elif fit_level == FitLevel.POOR:
        required_target = max(1, round(required_count * 0.25))
        max_unrelated = 5
    else:
        required_target = 0
        max_unrelated = 6

    return FitConstraint(
        target_overlap_min=overlap_min,
        target_overlap_max=overlap_max,
        required_skill_count_target=required_target,
        max_unrelated_skills=max_unrelated,
    )


def render_constraint_block(constraint: FitConstraint) -> str:
    """Render a human-readable constraints block for prompt injection."""
    return (
        f"- Target overlap band: {constraint.target_overlap_min:.2f}-{constraint.target_overlap_max:.2f}\n"
        f"- Aim to include about {constraint.required_skill_count_target} required skills\n"
        f"- Include no more than {constraint.max_unrelated_skills} unrelated skills"
    )
