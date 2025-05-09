"""Efficiency analysis for benchmark data."""

import pandas as pd
import numpy as np
from press_it.benchmark.analysis.core import (
    add_categorical_columns,
    get_best_score_column,
)


def analyze_quality_vs_size(df, output_path=None):
    """Analyze relationship between quality scores and compression ratio.

    Args:
        df: DataFrame with benchmark data
        output_path: Path to save analysis results

    Returns:
        DataFrame: Analysis results
    """
    if len(df) == 0:
        print("No data to analyze")
        return None

    # Try to get the best score column
    try:
        score_col = get_best_score_column(df)
    except ValueError:
        print("No valid quality scores available for analysis")
        return None

    # Calculate compression ratio if not present
    if (
        "compression_ratio" not in df.columns
        and "original_size" in df.columns
        and "compressed_size" in df.columns
    ):
        df["compression_ratio"] = df.apply(
            lambda row: (
                row["original_size"] / row["compressed_size"]
                if row["compressed_size"] > 0
                else float("inf")
            ),
            axis=1,
        )

    # Make a copy and make sure we have quality bins
    analysis_data = add_categorical_columns(df)

    # Group by compression type and quality level
    grouped = analysis_data.groupby(["compression_type", "quality"])

    # Aggregate compression metrics
    analysis_df = grouped.agg(
        {
            score_col: ["mean", "median", "std", "count"],
            "compression_ratio": ["mean", "median", "min", "max"],
            "original_size": ["mean"],
            "compressed_size": ["mean"],
        }
    ).reset_index()

    # Flatten MultiIndex columns
    analysis_df.columns = [
        "_".join(col).strip("_") for col in analysis_df.columns.values
    ]

    # Calculate coefficient of variation for score reliability
    analysis_df[f"score_cv"] = (
        analysis_df[f"{score_col}_std"] / analysis_df[f"{score_col}_mean"]
    )

    # Rename for clarity
    rename_dict = {
        f"{score_col}_mean": "best_score",
        f"{score_col}_count": "samples",
        "compression_ratio_mean": "compression_ratio",
    }
    analysis_df.rename(columns=rename_dict, inplace=True)

    # Filter for minimum sample size
    min_samples = 1
    analysis_df = analysis_df[analysis_df["samples"] >= min_samples]

    # Sort by compression type and quality
    analysis_df = analysis_df.sort_values(["compression_type", "best_score"])

    # Save if requested
    if output_path:
        analysis_df.to_csv(output_path, index=False)
        print(f"Saved quality vs. size analysis to {output_path}")

    return analysis_df


def analyze_efficiency_by_quality(df):
    """Determine the most efficient compression format at different quality levels.

    Args:
        df: DataFrame with benchmark data

    Returns:
        dict: Mapping of quality ranges to most efficient format
    """
    if "quality_bin" not in df.columns or len(df) == 0:
        # Create quality bins if not already present
        df = add_categorical_columns(df)

    if "quality_bin" not in df.columns or len(df) == 0:
        return {}

    # Ensure compression_ratio is available
    if "compression_ratio" not in df.columns:
        # Calculate it if we have the component values
        if "original_size" in df.columns and "compressed_size" in df.columns:
            df["compression_ratio"] = df.apply(
                lambda row: (
                    row["original_size"] / row["compressed_size"]
                    if row["compressed_size"] > 0
                    else float("inf")
                ),
                axis=1,
            )
        else:
            return {}

    # Find most efficient format for each quality bin
    best_formats = {}

    for quality_bin, group in df.groupby("quality_bin"):
        if len(group) < 2:
            continue

        # Get average compression ratio by format
        format_efficiency = group.groupby("compression_type")[
            "compression_ratio"
        ].mean()

        if len(format_efficiency) > 0:
            # Get format with highest compression ratio
            best_format = format_efficiency.idxmax()
            best_formats[str(quality_bin)] = best_format

    return best_formats


def analyze_image_factors(df):
    """Analyze how image characteristics affect compression efficiency.

    Args:
        df: DataFrame with benchmark data

    Returns:
        dict: Factor analysis results
    """
    if len(df) < 5:
        return {}

    # Make sure we have categorical columns and compression_ratio
    analysis_data = add_categorical_columns(df)

    # Ensure compression_ratio is calculated
    if "compression_ratio" not in analysis_data.columns:
        if (
            "original_size" in analysis_data.columns
            and "compressed_size" in analysis_data.columns
        ):
            analysis_data["compression_ratio"] = analysis_data.apply(
                lambda row: (
                    row["original_size"] / row["compressed_size"]
                    if row["compressed_size"] > 0
                    else float("inf")
                ),
                axis=1,
            )
        else:
            return {}

    # Analyze by size category
    if "size_category" not in analysis_data.columns:
        return {}

    size_analysis = (
        analysis_data.groupby(["size_category", "compression_type"])[
            "compression_ratio"
        ]
        .agg(["mean", "median", "count"])
        .reset_index()
    )

    # Convert to structured dictionary
    results = {"image_size_impact": {}}

    for size_cat, size_data in size_analysis.groupby("size_category"):
        if len(size_data) == 0 or size_data["count"].sum() < 3:
            continue

        # Find best format for this size category
        best_idx = size_data["mean"].idxmax()
        if pd.isna(best_idx):
            continue

        best_format = size_data.loc[best_idx]

        results["image_size_impact"][str(size_cat)] = {
            "best_format": best_format["compression_type"],
            "avg_compression": float(best_format["mean"]),
            "samples": int(best_format["count"]),
        }

    return results


def analyze_compression_vs_format(df):
    """Analyze compression performance across different formats.

    Args:
        df: DataFrame with benchmark data

    Returns:
        dict: Format comparison results
    """
    if len(df) < 5 or "compression_type" not in df.columns:
        return {}

    # Ensure compression_ratio is calculated
    if "compression_ratio" not in df.columns:
        if "original_size" in df.columns and "compressed_size" in df.columns:
            df["compression_ratio"] = df.apply(
                lambda row: (
                    row["original_size"] / row["compressed_size"]
                    if row["compressed_size"] > 0
                    else float("inf")
                ),
                axis=1,
            )
        else:
            return {}

    # Group by format and calculate statistics
    format_stats = (
        df.groupby("compression_type")["compression_ratio"]
        .agg(["mean", "median", "min", "max", "std", "count"])
        .reset_index()
    )

    # Convert to dictionary for easier access
    results = {}
    for _, row in format_stats.iterrows():
        format_name = row["compression_type"]
        results[format_name] = {
            "mean_compression": float(row["mean"]),
            "median_compression": float(row["median"]),
            "min_compression": float(row["min"]),
            "max_compression": float(row["max"]),
            "std_compression": float(row["std"]),
            "samples": int(row["count"]),
        }

    # Add rankings
    if results:
        # Rank by mean compression ratio
        ranked_formats = sorted(
            results.keys(), key=lambda x: results[x]["mean_compression"], reverse=True
        )
        for i, fmt in enumerate(ranked_formats):
            results[fmt]["rank"] = i + 1

    return results
