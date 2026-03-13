"""Tests for submission_bundle.sh grader artifact packager."""

from __future__ import annotations

import subprocess
from pathlib import Path


def _write(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _prepare_required_artifacts(project_root: Path) -> Path:
    # Copy script under test into isolated temp project root.
    source_script = Path(__file__).resolve().parent.parent / "submission_bundle.sh"
    script_path = project_root / "submission_bundle.sh"
    script_path.write_text(source_script.read_text(encoding="utf-8"), encoding="utf-8")

    # Required docs copied by script.
    _write(project_root / "README.md", "# temp readme\n")
    _write(project_root / "_submission_checklist.md", "# temp checklist\n")

    # Required generated artifacts.
    _write(project_root / "data/generated/jobs_20260101_000001.jsonl", "{}\n")
    _write(project_root / "data/generated/resumes_20260101_000001.jsonl", "{}\n")
    _write(project_root / "data/generated/pairs_20260101_000001.jsonl", "{}\n")

    # Required validation artifacts.
    _write(project_root / "outputs/validation/validated_data_20260101_000002.json", "{}\n")
    _write(project_root / "outputs/validation/invalid_20260101_000002.jsonl", "{}\n")
    _write(project_root / "outputs/validation/schema_failure_modes_20260101_000002.json", "{}\n")

    # Required analysis artifacts.
    _write(project_root / "outputs/analysis/failure_labels_20260101_000003.jsonl", "{}\n")
    _write(project_root / "outputs/analysis/failure_breakdown_20260101_000003.json", "{}\n")
    _write(project_root / "outputs/analysis/correlation_matrix_20260101_000003.json", "{}\n")

    # Required visualization artifacts.
    _write(project_root / "outputs/visualizations/failure_mode_correlation_heatmap_20260101_000004.png")
    _write(project_root / "outputs/visualizations/failure_rates_by_fit_level_20260101_000004.png")
    _write(project_root / "outputs/visualizations/failure_rates_by_template_20260101_000004.png")
    _write(project_root / "outputs/visualizations/niche_vs_standard_failure_20260101_000004.png")
    _write(project_root / "outputs/visualizations/schema_error_distribution_20260101_000004.png")

    # Required summary artifact.
    _write(project_root / "outputs/pipeline_summary_20260101_000005.json", "{}\n")
    return script_path


def test_submission_bundle_creates_bundle_dir_manifest_and_zip(tmp_path: Path) -> None:
    project_root = tmp_path / "mini-project-2"
    project_root.mkdir(parents=True, exist_ok=True)
    script_path = _prepare_required_artifacts(project_root)

    run_id = "test_run_001"
    subprocess.run(
        ["bash", str(script_path), run_id],
        cwd=project_root,
        check=True,
    )

    bundle_dir = project_root / "submission/mini-project-2" / run_id
    zip_path = project_root / "submission" / f"mini-project-2_{run_id}.zip"

    assert bundle_dir.exists()
    assert (bundle_dir / "MANIFEST.txt").exists()
    assert (bundle_dir / "README.md").exists()
    assert (bundle_dir / "_submission_checklist.md").exists()
    assert (bundle_dir / "failure_labels_20260101_000003.jsonl").exists()
    assert (bundle_dir / "pipeline_summary_20260101_000005.json").exists()
    assert zip_path.exists()


def test_submission_bundle_fails_when_required_artifact_missing(tmp_path: Path) -> None:
    project_root = tmp_path / "mini-project-2"
    project_root.mkdir(parents=True, exist_ok=True)
    script_path = _prepare_required_artifacts(project_root)

    # Remove one required file to trigger failure.
    (project_root / "outputs/analysis/failure_labels_20260101_000003.jsonl").unlink()

    result = subprocess.run(
        ["bash", str(script_path), "test_run_002"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Missing required artifact" in result.stderr
    assert "outputs/analysis/failure_labels_*.jsonl" in result.stderr
