"""Analysis of consistency between SSIMULACRA2 implementations."""

import pandas as pd
import numpy as np


def analyze_encoder_consistency(df):
    """Analyze consistency between different SSIMULACRA2 implementations.

    Args:
        df: DataFrame with benchmark data

    Returns:
        dict: Consistency statistics
    """
    consistency_stats = {}
    implementation_pairs = [("python", "cpp"), ("python", "rust"), ("cpp", "rust")]

    for impl1, impl2 in implementation_pairs:
        col1 = f"{impl1}_score"
        col2 = f"{impl2}_score"

        # Check if both columns exist and have data
        if col1 not in df.columns or col2 not in df.columns:
            continue

        # Get samples with both scores
        mask = df[col1].notna() & df[col2].notna()
        if mask.sum() < 5:  # Need minimum samples
            continue

        scores1 = df.loc[mask, col1]
        scores2 = df.loc[mask, col2]

        # Calculate correlation and differences
        correlation = scores1.corr(scores2)
        mean_diff = (scores1 - scores2).mean()
        abs_diff = (scores1 - scores2).abs()

        consistency_stats[f"{impl1}_vs_{impl2}"] = {
            "samples": int(mask.sum()),
            "correlation": float(correlation),
            "mean_difference": float(mean_diff),
            "max_difference": float(abs_diff.max()),
            "median_difference": float(abs_diff.median()),
        }

    return consistency_stats


def analyze_implementation_bias(df):
    """Analyze bias in different SSIMULACRA2 implementations.

    Args:
        df: DataFrame with benchmark data

    Returns:
        dict: Bias analysis results
    """
    implementations = ["python", "cpp", "rust"]

    # Check which implementations have data
    available_impls = [
        impl
        for impl in implementations
        if f"{impl}_score" in df.columns and df[f"{impl}_score"].notna().any()
    ]

    if len(available_impls) <= 1:
        return {}

    # Analyze bias by compression type
    results = {"by_format": {}}

    for fmt, group in df.groupby("compression_type"):
        format_results = {}

        for impl in available_impls:
            col = f"{impl}_score"
            if not group[col].notna().any():
                continue

            # Calculate average score
            avg_score = group[col].mean()
            format_results[impl] = float(avg_score)

        # Find implementation with highest/lowest average scores
        if len(format_results) > 1:
            highest_impl = max(format_results, key=format_results.get)
            lowest_impl = min(format_results, key=format_results.get)

            # Calculate max difference
            max_diff = format_results[highest_impl] - format_results[lowest_impl]

            results["by_format"][fmt] = {
                "scores": format_results,
                "highest_impl": highest_impl,
                "lowest_impl": lowest_impl,
                "max_difference": float(max_diff),
                "relative_difference": float(
                    max_diff / format_results[highest_impl] * 100
                ),
            }

    # Calculate overall bias
    all_scores = {}

    for impl in available_impls:
        col = f"{impl}_score"
        if not df[col].notna().any():
            continue

        all_scores[impl] = float(df[col].mean())

    if len(all_scores) > 1:
        highest_impl = max(all_scores, key=all_scores.get)
        lowest_impl = min(all_scores, key=all_scores.get)
        max_diff = all_scores[highest_impl] - all_scores[lowest_impl]

        results["overall"] = {
            "scores": all_scores,
            "highest_impl": highest_impl,
            "lowest_impl": lowest_impl,
            "max_difference": float(max_diff),
            "relative_difference": float(max_diff / all_scores[highest_impl] * 100),
        }

    return results


def analyze_implementation_reliability(df):
    """Analyze reliability of different SSIMULACRA2 implementations.

    Args:
        df: DataFrame with benchmark data

    Returns:
        dict: Reliability analysis results
    """
    implementations = ["python", "cpp", "rust"]

    # Check which implementations have data
    available_impls = [
        impl
        for impl in implementations
        if f"{impl}_score" in df.columns and df[f"{impl}_score"].notna().sum() >= 5
    ]

    if not available_impls:
        return {}

    results = {}

    # Calculate availability percentage
    total_samples = len(df)
    for impl in available_impls:
        col = f"{impl}_score"
        available_count = df[col].notna().sum()
        results[impl] = {
            "availability": float(available_count / total_samples * 100),
            "total_samples": int(total_samples),
            "available_samples": int(available_count),
        }

        # Calculate coefficient of variation for same input/format
        cv_values = []

        for fmt, group in df.groupby("compression_type"):
            if len(group) < 5 or not group[col].notna().any():
                continue

            # Group by quality setting
            for quality, quality_group in group.groupby("quality"):
                if len(quality_group) < 3 or not quality_group[col].notna().any():
                    continue

                # Calculate CV for this format/quality
                mean = quality_group[col].mean()
                std = quality_group[col].std()
                if mean > 0:
                    cv = std / mean
                    cv_values.append(cv)

        if cv_values:
            results[impl]["avg_cv"] = float(np.mean(cv_values))
            results[impl]["max_cv"] = float(np.max(cv_values))

    # Rank implementations by reliability (lower CV is better)
    if all("avg_cv" in results[impl] for impl in available_impls):
        ranked_impls = sorted(available_impls, key=lambda x: results[x]["avg_cv"])
        for i, impl in enumerate(ranked_impls):
            results[impl]["reliability_rank"] = i + 1

    return results
