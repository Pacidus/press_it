"""Quality analysis for benchmark data."""

import pandas as pd
import numpy as np
from press_it.benchmark.analysis.core import (
    add_categorical_columns,
    get_best_score_column,
)


def analyze_quality_distribution(df):
    """Analyze the distribution of quality scores across formats.

    Args:
        df: DataFrame with benchmark data

    Returns:
        dict: Quality distribution analysis
    """
    try:
        score_col = get_best_score_column(df)
    except ValueError:
        return {}

    # Make sure we have quality bins
    analysis_data = add_categorical_columns(df)

    # Count samples in each quality bin by format
    if (
        "quality_bin" not in analysis_data.columns
        or "compression_type" not in analysis_data.columns
    ):
        return {}

    quality_counts = (
        analysis_data.groupby(["compression_type", "quality_bin"])
        .size()
        .unstack(fill_value=0)
    )

    # Calculate percentage distribution
    quality_pct = quality_counts.div(quality_counts.sum(axis=1), axis=0) * 100

    # Convert to dictionary format
    result = {"quality_distribution": {}}

    for comp_type in quality_counts.index:
        result["quality_distribution"][comp_type] = {
            "counts": {
                str(bin_name): int(count)
                for bin_name, count in quality_counts.loc[comp_type].items()
            },
            "percentages": {
                str(bin_name): float(pct)
                for bin_name, pct in quality_pct.loc[comp_type].items()
            },
        }

    # Add overall quality stats
    result["overall_quality"] = {
        "mean": float(analysis_data[score_col].mean()),
        "median": float(analysis_data[score_col].median()),
        "min": float(analysis_data[score_col].min()),
        "max": float(analysis_data[score_col].max()),
    }

    return result


def analyze_quality_vs_settings(df):
    """Analyze relationship between quality setting and resulting quality score.

    Args:
        df: DataFrame with benchmark data

    Returns:
        dict: Analysis of quality settings vs scores
    """
    if len(df) < 5 or "quality" not in df.columns:
        return {}

    try:
        score_col = get_best_score_column(df)
    except ValueError:
        return {}

    # Group by format and quality setting
    results = {}

    for fmt, group in df.groupby("compression_type"):
        # Create groups of quality settings (e.g., 0-10, 10-20, ...)
        group["quality_group"] = pd.cut(
            group["quality"],
            bins=range(0, 101, 10),
            labels=[f"{i}-{i+10}" for i in range(0, 100, 10)],
        )

        # Calculate statistics for each quality group
        quality_stats = (
            group.groupby("quality_group")[score_col]
            .agg(["mean", "median", "std", "min", "max", "count"])
            .reset_index()
        )

        # Convert to dictionary
        fmt_data = {}
        for _, row in quality_stats.iterrows():
            if pd.isna(row["mean"]):
                continue

            fmt_data[str(row["quality_group"])] = {
                "mean_score": float(row["mean"]),
                "median_score": float(row["median"]),
                "min_score": float(row["min"]),
                "max_score": float(row["max"]),
                "std_score": float(row["std"]),
                "samples": int(row["count"]),
            }

        if fmt_data:
            results[fmt] = fmt_data

    return results


def analyze_quality_consistency(df):
    """Analyze how consistent quality is within the same format and settings.

    Args:
        df: DataFrame with benchmark data

    Returns:
        dict: Quality consistency analysis
    """
    if len(df) < 10:
        return {}

    try:
        score_col = get_best_score_column(df)
    except ValueError:
        return {}

    # Group by format
    results = {}

    for fmt, group in df.groupby("compression_type"):
        # Calculate coefficient of variation for each quality level
        if "quality" not in group.columns:
            continue

        cv_data = (
            group.groupby("quality")[score_col]
            .agg(["mean", "std", "count"])
            .reset_index()
        )

        # Calculate coefficient of variation (CV)
        cv_data["cv"] = cv_data["std"] / cv_data["mean"]

        # Keep only groups with enough samples
        cv_data = cv_data[cv_data["count"] >= 3]

        if len(cv_data) > 0:
            # Get average CV for this format
            avg_cv = cv_data["cv"].mean()

            # Get most consistent quality level (lowest CV)
            most_consistent_idx = cv_data["cv"].idxmin()
            most_consistent = cv_data.loc[most_consistent_idx]

            # Get least consistent quality level (highest CV)
            least_consistent_idx = cv_data["cv"].idxmax()
            least_consistent = cv_data.loc[least_consistent_idx]

            # Add to results
            results[fmt] = {
                "average_cv": float(avg_cv),
                "most_consistent": {
                    "quality": int(most_consistent["quality"]),
                    "cv": float(most_consistent["cv"]),
                    "mean_score": float(most_consistent["mean"]),
                    "samples": int(most_consistent["count"]),
                },
                "least_consistent": {
                    "quality": int(least_consistent["quality"]),
                    "cv": float(least_consistent["cv"]),
                    "mean_score": float(least_consistent["mean"]),
                    "samples": int(least_consistent["count"]),
                },
            }

    # Add overall rankings by consistency
    if results:
        # Sort formats by average CV (lower is better)
        formats_by_consistency = sorted(
            results.keys(), key=lambda x: results[x]["average_cv"]
        )

        for i, fmt in enumerate(formats_by_consistency):
            results[fmt]["consistency_rank"] = i + 1

    return results


def analyze_quality_thresholds(df):
    """Analyze quality thresholds for different formats.

    Args:
        df: DataFrame with benchmark data

    Returns:
        dict: Quality threshold analysis
    """
    try:
        score_col = get_best_score_column(df)
    except ValueError:
        return {}

    # Standard quality thresholds
    thresholds = [60, 70, 80, 90, 95]

    results = {}

    for fmt, group in df.groupby("compression_type"):
        # Sort by quality
        sorted_data = group.sort_values("quality")

        # Skip if too few samples
        if len(sorted_data) < 5:
            continue

        threshold_data = {}

        for threshold in thresholds:
            # Find samples that meet or exceed the threshold
            meets_threshold = sorted_data[sorted_data[score_col] >= threshold]

            if len(meets_threshold) > 0:
                # Get the lowest quality setting that meets the threshold
                min_quality = meets_threshold["quality"].min()

                # Get the corresponding row(s)
                threshold_rows = meets_threshold[
                    meets_threshold["quality"] == min_quality
                ]

                # Get compression ratio
                avg_ratio = threshold_rows["compression_ratio"].mean()

                # Add to results
                threshold_data[str(threshold)] = {
                    "min_quality_setting": int(min_quality),
                    "avg_compression_ratio": float(avg_ratio),
                    "samples": len(threshold_rows),
                }

        if threshold_data:
            results[fmt] = threshold_data

    return results
