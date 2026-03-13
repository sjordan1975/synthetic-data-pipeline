"""Phase 6 visualization pipeline for analysis artifacts.

Generates rubric-required charts from Phase 4/5 outputs, exports chart source
tables as CSV, and writes concise interpretation notes.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


FAILURE_FLAG_COLUMNS: tuple[str, ...] = (
    "experience_mismatch",
    "seniority_mismatch",
    "missing_core_skills",
    "hallucinated_skills",
    "awkward_language",
)


def find_latest_file(directory: Path, pattern: str) -> Path | None:
    """Return newest file by mtime for a pattern within directory."""
    matches = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not matches:
        return None
    return matches[-1]


def read_json(path: Path) -> dict[str, Any]:
    """Read JSON object from file."""
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL rows into list of dictionaries."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_artifacts(
    analysis_dir: Path,
    validation_dir: Path,
    validated_data_json: Path | None,
) -> dict[str, Path]:
    """Resolve latest artifact paths required for visualization."""
    failure_labels = find_latest_file(analysis_dir, "failure_labels_*.jsonl")
    correlation = find_latest_file(analysis_dir, "correlation_matrix_*.json")
    breakdown = find_latest_file(analysis_dir, "failure_breakdown_*.json")
    schema_failure_modes = find_latest_file(validation_dir, "schema_failure_modes_*.json")

    if validated_data_json is None:
        validated_data_json = find_latest_file(validation_dir, "validated_data_*.json")

    required = {
        "failure_labels": failure_labels,
        "correlation": correlation,
        "breakdown": breakdown,
        "schema_failure_modes": schema_failure_modes,
        "validated_data": validated_data_json,
    }

    missing = [name for name, path in required.items() if path is None]
    if missing:
        raise FileNotFoundError(
            "Missing required visualization inputs: " + ", ".join(sorted(missing))
        )

    return {name: path for name, path in required.items() if path is not None}


def build_fit_level_rates_df(breakdown_payload: dict[str, Any]) -> pd.DataFrame:
    """Convert failure breakdown by fit level to plotting dataframe."""
    by_fit_level = breakdown_payload.get("by_fit_level", {})
    rows: list[dict[str, Any]] = []
    for fit_level, metrics in by_fit_level.items():
        row = {
            "fit_level": fit_level,
            "count": int(metrics.get("count", 0)),
            "mean_overlap": float(metrics.get("mean_overlap", 0.0)),
            "mean_failure_count": float(metrics.get("mean_failure_count", 0.0)),
        }
        for col in FAILURE_FLAG_COLUMNS:
            row[f"{col}_rate"] = float(metrics.get(f"{col}_rate", 0.0))
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df["fit_level"] = pd.Categorical(
            df["fit_level"],
            categories=["excellent", "good", "partial", "poor", "complete_mismatch", "unlabeled"],
            ordered=True,
        )
        df = df.sort_values("fit_level")
    return df


def build_template_rates_df(breakdown_payload: dict[str, Any]) -> pd.DataFrame:
    """Convert failure breakdown by template to dataframe."""
    by_template = breakdown_payload.get("by_template", {})
    rows: list[dict[str, Any]] = []
    for template, metrics in by_template.items():
        rows.append(
            {
                "resume_template": template,
                "count": int(metrics.get("count", 0)),
                "mean_overlap": float(metrics.get("mean_overlap", 0.0)),
                "mean_failure_count": float(metrics.get("mean_failure_count", 0.0)),
                "awkward_language_rate": float(metrics.get("awkward_language_rate", 0.0)),
                "hallucinated_skills_rate": float(metrics.get("hallucinated_skills_rate", 0.0)),
            }
        )
    return pd.DataFrame(rows)


def build_correlation_df(correlation_payload: dict[str, Any]) -> pd.DataFrame:
    """Build square correlation dataframe for heatmap plotting."""
    matrix = correlation_payload.get("matrix", {})
    df = pd.DataFrame(matrix)
    if not df.empty:
        df = df.reindex(index=df.columns)
    return df


def build_schema_error_df(schema_failure_payload: dict[str, Any]) -> pd.DataFrame:
    """Build schema error distribution dataframe."""
    distribution = schema_failure_payload.get("error_type_distribution", {})
    rows = [
        {"error_type": key, "count": int(value)}
        for key, value in distribution.items()
    ]
    return pd.DataFrame(rows)


def build_niche_mapping_df(validated_data_payload: dict[str, Any]) -> pd.DataFrame:
    """Extract trace_id -> is_niche_role mapping from validated pairs payload."""
    records = validated_data_payload.get("records", {})
    pairs = records.get("pairs", [])

    rows: list[dict[str, Any]] = []
    for pair in pairs:
        trace_id = pair.get("trace_id")
        job = pair.get("job_description", {})
        metadata = job.get("metadata", {})
        is_niche = bool(metadata.get("is_niche_role", False))
        rows.append({"trace_id": trace_id, "is_niche_role": is_niche})

    return pd.DataFrame(rows)


def build_niche_vs_standard_df(
    failure_labels_df: pd.DataFrame,
    niche_mapping_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build grouped failure rates comparing niche vs standard roles."""
    merged = failure_labels_df.merge(niche_mapping_df, on="trace_id", how="left")
    merged["is_niche_role"] = merged["is_niche_role"].fillna(False)
    merged["role_type"] = merged["is_niche_role"].map({True: "niche", False: "standard"})

    rows: list[dict[str, Any]] = []
    for role_type, frame in merged.groupby("role_type"):
        row: dict[str, Any] = {
            "role_type": role_type,
            "count": int(len(frame)),
            "mean_overlap": float(frame["skills_overlap"].mean()),
            "mean_failure_count": float(frame["failure_count"].mean()),
        }
        for metric in FAILURE_FLAG_COLUMNS:
            row[f"{metric}_rate"] = float(frame[metric].astype(float).mean())
        rows.append(row)

    return pd.DataFrame(rows)


def apply_plot_style() -> None:
    """Set consistent plotting style for all charts."""
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.dpi"] = 120


def save_chart(fig: plt.Figure, output_path: Path) -> None:
    """Save chart with consistent layout and quality."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    """Save chart source data to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def chart_correlation_heatmap(df: pd.DataFrame, png_path: Path) -> None:
    """Render failure-mode correlation heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        df,
        cmap="coolwarm",
        center=0.0,
        annot=True,
        fmt=".2f",
        linewidths=0.4,
        cbar_kws={"label": "Pearson correlation"},
        ax=ax,
    )
    ax.set_title("Failure Mode Correlation Heatmap")
    save_chart(fig, png_path)


def chart_failure_rates_by_fit_level(df: pd.DataFrame, png_path: Path) -> None:
    """Render grouped failure-rate bar chart by fit level."""
    rate_cols = [f"{metric}_rate" for metric in FAILURE_FLAG_COLUMNS]
    plot_df = df[["fit_level", "count", *rate_cols]].melt(
        id_vars=["fit_level", "count"],
        value_vars=rate_cols,
        var_name="metric",
        value_name="rate",
    )
    plot_df["metric"] = plot_df["metric"].str.replace("_rate", "", regex=False)

    fig, ax = plt.subplots(figsize=(13, 6))
    sns.barplot(data=plot_df, x="fit_level", y="rate", hue="metric", ax=ax)
    ax.set_ylim(0.0, 1.08)
    ax.set_xlabel("Generated fit level")
    ax.set_ylabel("Failure rate")
    ax.set_title("Failure Rates by Fit Level", pad=18)
    ax.tick_params(axis="x", rotation=18)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    ax.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")

    for idx, row in df.reset_index(drop=True).iterrows():
        ax.text(idx, 1.01, f"n={int(row['count'])}", ha="center", va="bottom", fontsize=9)

    save_chart(fig, png_path)


def chart_failure_rates_by_template(df: pd.DataFrame, png_path: Path) -> None:
    """Render grouped chart for template-level failure rates."""
    rate_cols = ["awkward_language_rate", "hallucinated_skills_rate"]
    plot_df = df[["resume_template", "count", *rate_cols]].melt(
        id_vars=["resume_template", "count"],
        value_vars=rate_cols,
        var_name="metric",
        value_name="rate",
    )
    plot_df["metric"] = plot_df["metric"].str.replace("_rate", "", regex=False)

    fig, ax = plt.subplots(figsize=(13, 6))
    sns.barplot(data=plot_df, x="resume_template", y="rate", hue="metric", ax=ax)
    ax.set_ylim(0.0, 1.08)
    ax.set_xlabel("Resume template")
    ax.set_ylabel("Failure rate")
    ax.set_title("Failure Rates by Template", pad=18)
    ax.tick_params(axis="x", rotation=18)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    ax.legend(title="Metric")

    for idx, row in df.reset_index(drop=True).iterrows():
        ax.text(idx, 1.01, f"n={int(row['count'])}", ha="center", va="bottom", fontsize=9)

    save_chart(fig, png_path)


def chart_niche_vs_standard(df: pd.DataFrame, png_path: Path) -> None:
    """Render niche vs standard role failure comparison."""
    rate_cols = [f"{metric}_rate" for metric in FAILURE_FLAG_COLUMNS]
    plot_df = df[["role_type", "count", *rate_cols]].melt(
        id_vars=["role_type", "count"],
        value_vars=rate_cols,
        var_name="metric",
        value_name="rate",
    )
    plot_df["metric"] = plot_df["metric"].str.replace("_rate", "", regex=False)

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=plot_df, x="role_type", y="rate", hue="metric", ax=ax)
    ax.set_ylim(0.0, 1.08)
    ax.set_xlabel("Role type")
    ax.set_ylabel("Failure rate")
    ax.set_title("Niche vs Standard Role Failure Comparison", pad=18)
    ax.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")

    for idx, row in df.reset_index(drop=True).iterrows():
        ax.text(idx, 1.01, f"n={int(row['count'])}", ha="center", va="bottom", fontsize=9)

    save_chart(fig, png_path)


def chart_schema_error_distribution(df: pd.DataFrame, png_path: Path) -> None:
    """Render schema validation error distribution chart."""
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df, x="count", y="error_type", color="#3b82f6", ax=ax)
    ax.set_xlabel("Count")
    ax.set_ylabel("Validation error type")
    ax.set_title("Schema Validation Error Distribution")

    for patch in ax.patches:
        width = patch.get_width()
        y = patch.get_y() + patch.get_height() / 2
        ax.text(width + 0.05, y, f"{int(width)}", va="center", fontsize=9)

    save_chart(fig, png_path)


def build_interpretation_notes(
    fit_df: pd.DataFrame,
    template_df: pd.DataFrame,
    niche_df: pd.DataFrame,
    schema_df: pd.DataFrame,
    correlation_df: pd.DataFrame,
) -> list[str]:
    """Generate concise, chart-level interpretation notes."""
    notes: list[str] = []

    if not correlation_df.empty:
        matrix = correlation_df.copy(deep=True)
        diagonal_len = min(matrix.shape[0], matrix.shape[1])
        for idx in range(diagonal_len):
            matrix.iat[idx, idx] = 0.0
        strongest = matrix.stack().abs().sort_values(ascending=False)
        if not strongest.empty:
            (metric_a, metric_b), magnitude = strongest.index[0], strongest.iloc[0]
            notes.append(
                f"- **Correlation heatmap:** The strongest pairwise relationship is between `{metric_a}` and `{metric_b}` (|r|={magnitude:.2f}), suggesting these failure modes often move together."
            )

    if not fit_df.empty:
        worst_fit = fit_df.sort_values("mean_failure_count", ascending=False).iloc[0]
        best_fit = fit_df.sort_values("mean_failure_count", ascending=True).iloc[0]
        notes.append(
            f"- **Failure rates by fit level:** `{worst_fit['fit_level']}` has the highest mean failure count ({worst_fit['mean_failure_count']:.2f}), while `{best_fit['fit_level']}` has the lowest ({best_fit['mean_failure_count']:.2f})."
        )

    if not template_df.empty:
        top_template = template_df.sort_values("mean_failure_count", ascending=False).iloc[0]
        notes.append(
            f"- **Failure rates by template:** `{top_template['resume_template']}` currently shows mean failure count {top_template['mean_failure_count']:.2f}; use this as baseline for future template comparisons."
        )

    if not niche_df.empty and niche_df["role_type"].nunique() > 1:
        grouped = niche_df.set_index("role_type")
        if {"niche", "standard"}.issubset(grouped.index):
            diff = grouped.loc["niche", "mean_failure_count"] - grouped.loc["standard", "mean_failure_count"]
            notes.append(
                f"- **Niche vs standard:** Niche roles differ from standard roles by {diff:+.2f} in mean failure count, indicating role specialization impacts failure profile." 
            )

    if not schema_df.empty:
        top_error = schema_df.sort_values("count", ascending=False).iloc[0]
        notes.append(
            f"- **Schema error distribution:** Most frequent schema issue is `{top_error['error_type']}` (count={int(top_error['count'])})."
        )

    return notes


def run_visualization(
    analysis_dir: Path,
    validation_dir: Path,
    output_dir: Path,
    validated_data_json: Path | None,
) -> dict[str, Path]:
    """Run full Phase 6 visualization pipeline."""
    apply_plot_style()

    artifact_paths = load_artifacts(
        analysis_dir=analysis_dir,
        validation_dir=validation_dir,
        validated_data_json=validated_data_json,
    )

    failure_labels_df = pd.DataFrame(read_jsonl(artifact_paths["failure_labels"]))
    breakdown_payload = read_json(artifact_paths["breakdown"])
    correlation_payload = read_json(artifact_paths["correlation"])
    schema_payload = read_json(artifact_paths["schema_failure_modes"])
    validated_payload = read_json(artifact_paths["validated_data"])

    fit_df = build_fit_level_rates_df(breakdown_payload)
    template_df = build_template_rates_df(breakdown_payload)
    correlation_df = build_correlation_df(correlation_payload)
    schema_df = build_schema_error_df(schema_payload)
    niche_map_df = build_niche_mapping_df(validated_payload)
    niche_df = build_niche_vs_standard_df(failure_labels_df, niche_map_df)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    files: dict[str, Path] = {
        "correlation_png": output_dir / f"failure_mode_correlation_heatmap_{timestamp}.png",
        "correlation_csv": output_dir / f"failure_mode_correlation_heatmap_{timestamp}.csv",
        "fit_level_png": output_dir / f"failure_rates_by_fit_level_{timestamp}.png",
        "fit_level_csv": output_dir / f"failure_rates_by_fit_level_{timestamp}.csv",
        "template_png": output_dir / f"failure_rates_by_template_{timestamp}.png",
        "template_csv": output_dir / f"failure_rates_by_template_{timestamp}.csv",
        "niche_png": output_dir / f"niche_vs_standard_failure_{timestamp}.png",
        "niche_csv": output_dir / f"niche_vs_standard_failure_{timestamp}.csv",
        "schema_png": output_dir / f"schema_error_distribution_{timestamp}.png",
        "schema_csv": output_dir / f"schema_error_distribution_{timestamp}.csv",
        "notes": output_dir / f"interpretation_notes_{timestamp}.md",
    }

    chart_correlation_heatmap(correlation_df, files["correlation_png"])
    chart_failure_rates_by_fit_level(fit_df, files["fit_level_png"])
    chart_failure_rates_by_template(template_df, files["template_png"])
    chart_niche_vs_standard(niche_df, files["niche_png"])
    chart_schema_error_distribution(schema_df, files["schema_png"])

    save_csv(correlation_df.reset_index(names="metric"), files["correlation_csv"])
    save_csv(fit_df, files["fit_level_csv"])
    save_csv(template_df, files["template_csv"])
    save_csv(niche_df, files["niche_csv"])
    save_csv(schema_df, files["schema_csv"])

    notes = build_interpretation_notes(
        fit_df=fit_df,
        template_df=template_df,
        niche_df=niche_df,
        schema_df=schema_df,
        correlation_df=correlation_df,
    )
    notes_text = "# Phase 6 Interpretation Notes\n\n" + "\n".join(notes)
    files["notes"].write_text(notes_text + "\n", encoding="utf-8")

    return files


def parse_args() -> argparse.Namespace:
    """Parse CLI args for visualization script."""
    parser = argparse.ArgumentParser(description="Phase 6 visualization pipeline")
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("outputs/analysis"),
        help="Directory with analyzer outputs.",
    )
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=Path("outputs/validation"),
        help="Directory with validator outputs.",
    )
    parser.add_argument(
        "--validated-data-json",
        type=Path,
        default=None,
        help="Optional explicit validated_data JSON path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/visualizations"),
        help="Directory for visualization PNG/CSV outputs.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for visualization generation."""
    args = parse_args()
    files = run_visualization(
        analysis_dir=args.analysis_dir,
        validation_dir=args.validation_dir,
        output_dir=args.output_dir,
        validated_data_json=args.validated_data_json,
    )

    print("=" * 72)
    print("PHASE 6 - VISUALIZATION")
    print("=" * 72)
    print("Saved visualization artifacts:")
    for key, path in files.items():
        print(f"- {key}: {path}")


if __name__ == "__main__":
    main()
