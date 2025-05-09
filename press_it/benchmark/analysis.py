"""Analysis utilities for press_it benchmark data."""

import os
import glob
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict


def get_benchmark_files(results_dir=None):
    """Find all benchmark Parquet files in the results directory.

    Args:
        results_dir (str, optional): Directory to search in. Default is current directory.

    Returns:
        list: Sorted list of benchmark file paths
    """
    search_dir = Path(results_dir or ".")
    benchmark_files = []

    # Look for parquet files
    for pattern in ["*benchmark*.parquet", "*ssimulacra2*.parquet"]:
        benchmark_files.extend(search_dir.glob(pattern))

    # Sort by modification time (newest last)
    benchmark_files.sort(key=lambda x: os.path.getmtime(x))
    return benchmark_files


def load_benchmark_data(file_path):
    """Load benchmark data from a Parquet file.

    Args:
        file_path (str): Path to benchmark Parquet file

    Returns:
        pd.DataFrame: Loaded benchmark data or None if loading fails
    """
    try:
        df = pd.read_parquet(file_path)
        print(f"Loaded {len(df)} rows from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading benchmark data: {e}")
        return None


def combine_benchmark_files(file_paths, output_path=None):
    """Combine multiple benchmark files into a single DataFrame.

    Args:
        file_paths (list): List of benchmark file paths
        output_path (str, optional): Path to save combined data

    Returns:
        pd.DataFrame: Combined benchmark data
    """
    combined_df = pd.DataFrame()

    for file_path in file_paths:
        try:
            df = pd.read_parquet(file_path)
            print(f"Adding {len(df)} rows from {file_path}")
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    # Deduplicate based on file paths (original_path, compressed_path)
    if len(combined_df) > 0:
        before_count = len(combined_df)
        combined_df = combined_df.drop_duplicates(
            subset=["original_path", "compressed_path"]
        )
        after_count = len(combined_df)
        if before_count > after_count:
            print(f"Removed {before_count - after_count} duplicate entries")

        # Sort by timestamp if available
        if "timestamp" in combined_df.columns:
            combined_df = combined_df.sort_values("timestamp")

        # Save combined data if requested
        if output_path:
            combined_df.to_parquet(
                output_path, engine="pyarrow", compression="snappy", index=False
            )
            print(f"Saved combined data ({len(combined_df)} rows) to {output_path}")

    return combined_df


def summarize_benchmark_data(df):
    """Generate summary statistics from benchmark data.

    Args:
        df (pd.DataFrame): Benchmark data

    Returns:
        dict: Summary statistics
    """
    summary = {
        "total_samples": len(df),
        "compression_types": {},
        "image_dimensions": {},
        "quality_range": {},
        "score_stats": {},
        "compression_stats": {},
        "implementation_availability": {
            "python": (
                (df["python_score"].notna().sum() / len(df)) * 100 if len(df) > 0 else 0
            ),
            "cpp": (
                (df["cpp_score"].notna().sum() / len(df)) * 100 if len(df) > 0 else 0
            ),
            "rust": (
                (df["rust_score"].notna().sum() / len(df)) * 100 if len(df) > 0 else 0
            ),
        },
    }

    # Handle empty dataframe
    if len(df) == 0:
        return summary

    # Count by compression type
    compression_counts = df["compression_type"].value_counts().to_dict()
    summary["compression_types"] = {
        comp_type: {
            "count": count,
            "percentage": (count / len(df)) * 100,
        }
        for comp_type, count in compression_counts.items()
    }

    # Image dimensions stats
    summary["image_dimensions"] = {
        "width": {
            "min": df["width"].min(),
            "max": df["width"].max(),
            "mean": round(df["width"].mean(), 2),
            "median": df["width"].median(),
        },
        "height": {
            "min": df["height"].min(),
            "max": df["height"].max(),
            "mean": round(df["height"].mean(), 2),
            "median": df["height"].median(),
        },
    }

    # Quality settings range by compression type
    summary["quality_range"] = {
        comp_type: {
            "min": group["quality"].min(),
            "max": group["quality"].max(),
            "mean": round(group["quality"].mean(), 2),
        }
        for comp_type, group in df.groupby("compression_type")
    }

    # SSIMULACRA2 score statistics
    # Use the Python implementation as reference as it's most likely to be available
    if df["python_score"].notna().any():
        summary["score_stats"]["overall"] = {
            "min": round(df["python_score"].min(), 2),
            "max": round(df["python_score"].max(), 2),
            "mean": round(df["python_score"].mean(), 2),
            "median": round(df["python_score"].median(), 2),
            "std": round(df["python_score"].std(), 2),
        }

        # Score stats by compression type
        summary["score_stats"]["by_compression_type"] = {
            comp_type: {
                "min": round(group["python_score"].min(), 2),
                "max": round(group["python_score"].max(), 2),
                "mean": round(group["python_score"].mean(), 2),
                "median": round(group["python_score"].median(), 2),
            }
            for comp_type, group in df.groupby("compression_type")
            if group["python_score"].notna().any()
        }

    # Compression ratio statistics
    summary["compression_stats"]["overall"] = {
        "min": round(df["compression_ratio"].min(), 2),
        "max": round(df["compression_ratio"].max(), 2),
        "mean": round(df["compression_ratio"].mean(), 2),
        "median": round(df["compression_ratio"].median(), 2),
    }

    # Compression stats by format
    summary["compression_stats"]["by_compression_type"] = {
        comp_type: {
            "min": round(group["compression_ratio"].min(), 2),
            "max": round(group["compression_ratio"].max(), 2),
            "mean": round(group["compression_ratio"].mean(), 2),
            "median": round(group["compression_ratio"].median(), 2),
        }
        for comp_type, group in df.groupby("compression_type")
    }

    # Add most efficient format at different quality levels
    summary["most_efficient_format"] = analyze_efficiency_by_quality(df)

    return summary


def print_summary(summary):
    """Print a human-readable summary of benchmark results.

    Args:
        summary (dict): Summary statistics from summarize_benchmark_data
    """
    print("\n===== BENCHMARK SUMMARY =====")
    print(f"Total samples: {summary['total_samples']}")

    # Print compression type breakdown
    print("\n--- Compression Type Distribution ---")
    for comp_type, stats in summary["compression_types"].items():
        print(f"{comp_type}: {stats['count']} samples ({stats['percentage']:.2f}%)")

    # Print score statistics if available
    if "score_stats" in summary and "overall" in summary["score_stats"]:
        print("\n--- SSIMULACRA2 Score Statistics ---")
        print(
            f"Overall: min={summary['score_stats']['overall']['min']:.2f}, "
            f"max={summary['score_stats']['overall']['max']:.2f}, "
            f"mean={summary['score_stats']['overall']['mean']:.2f}, "
            f"median={summary['score_stats']['overall']['median']:.2f}"
        )

        if "by_compression_type" in summary["score_stats"]:
            print("\nBy compression type:")
            for comp_type, stats in summary["score_stats"][
                "by_compression_type"
            ].items():
                print(
                    f"  {comp_type}: min={stats['min']:.2f}, "
                    f"max={stats['max']:.2f}, "
                    f"mean={stats['mean']:.2f}, "
                    f"median={stats['median']:.2f}"
                )

    # Print compression statistics
    print("\n--- Compression Ratio Statistics ---")
    print(
        f"Overall: min={summary['compression_stats']['overall']['min']:.2f}x, "
        f"max={summary['compression_stats']['overall']['max']:.2f}x, "
        f"mean={summary['compression_stats']['overall']['mean']:.2f}x, "
        f"median={summary['compression_stats']['overall']['median']:.2f}x"
    )

    print("\nBy compression type:")
    for comp_type, stats in summary["compression_stats"]["by_compression_type"].items():
        print(
            f"  {comp_type}: min={stats['min']:.2f}x, "
            f"max={stats['max']:.2f}x, "
            f"mean={stats['mean']:.2f}x, "
            f"median={stats['median']:.2f}x"
        )

    # Print most efficient formats by quality level
    if "most_efficient_format" in summary:
        print("\n--- Most Efficient Format by Quality Level ---")
        for quality_range, format_name in summary["most_efficient_format"].items():
            print(f"Quality {quality_range}: {format_name}")

    print("\n--- Implementation Availability ---")
    for impl, percentage in summary["implementation_availability"].items():
        print(f"{impl}: {percentage:.2f}%")


def analyze_quality_vs_size(df, output_path=None):
    """Analyze relationship between quality scores and compression ratio.

    Args:
        df (pd.DataFrame): Benchmark data
        output_path (str, optional): Path to save analysis results

    Returns:
        pd.DataFrame: Analysis results
    """
    if len(df) == 0:
        print("No data to analyze")
        return None

    # Use Python scores as they're most likely to be available
    if not df["python_score"].notna().any():
        print("No valid quality scores available for analysis")
        return None

    # Make a copy to avoid modifying the original dataframe
    analysis_data = df.copy()

    # Group by compression type and quality level
    grouped = analysis_data.groupby(["compression_type", "quality"])

    # Aggregate compression metrics
    analysis_df = grouped.agg(
        {
            "python_score": ["mean", "median", "std", "count"],
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
    analysis_df["score_cv"] = (
        analysis_df["python_score_std"] / analysis_df["python_score_mean"]
    )

    # Rename for clarity
    analysis_df.rename(
        columns={
            "python_score_mean": "best_score",
            "python_score_count": "samples",
            "compression_ratio_mean": "compression_ratio",
        },
        inplace=True,
    )

    # Filter to ensure minimum sample size
    min_samples = 1
    analysis_df = analysis_df[analysis_df["samples"] >= min_samples]

    # Sort by compression type and quality score
    analysis_df = analysis_df.sort_values(["compression_type", "best_score"])

    if output_path:
        analysis_df.to_csv(output_path, index=False)
        print(f"Saved quality vs. size analysis to {output_path}")

    return analysis_df


def analyze_efficiency_by_quality(df):
    """Determine the most efficient compression format at different quality levels.

    Args:
        df (pd.DataFrame): Benchmark data

    Returns:
        dict: Mapping of quality ranges to most efficient format
    """
    if len(df) == 0 or not df["python_score"].notna().any():
        return {}

    # Make a copy to avoid modifying the original dataframe
    analysis_data = df.copy()

    # Create quality bins
    quality_bins = [0, 60, 70, 80, 90, 100]
    analysis_data["quality_range"] = pd.cut(
        analysis_data["python_score"],
        bins=quality_bins,
        labels=[f"{q1}-{q2}" for q1, q2 in zip(quality_bins[:-1], quality_bins[1:])],
    )

    # Find the most efficient format for each quality range
    best_formats = {}
    for quality_range, group in analysis_data.groupby("quality_range"):
        if len(group) < 2:
            continue

        # Group by compression type and find average compression ratio
        format_efficiency = group.groupby("compression_type")[
            "compression_ratio"
        ].mean()

        if len(format_efficiency) > 0:
            best_format = format_efficiency.idxmax()
            best_formats[str(quality_range)] = best_format

    return best_formats


def analyze_encoder_consistency(df):
    """Analyze consistency between different SSIMULACRA2 implementations.

    Args:
        df (pd.DataFrame): Benchmark data

    Returns:
        dict: Consistency statistics
    """
    consistency_stats = {}
    implementation_pairs = [("python", "cpp"), ("python", "rust"), ("cpp", "rust")]

    for impl1, impl2 in implementation_pairs:
        # Filter for samples with both implementations available
        mask = df[f"{impl1}_score"].notna() & df[f"{impl2}_score"].notna()
        if mask.sum() < 5:  # Need minimum samples for meaningful comparison
            continue

        scores1 = df.loc[mask, f"{impl1}_score"]
        scores2 = df.loc[mask, f"{impl2}_score"]

        # Calculate correlation and difference statistics
        correlation = scores1.corr(scores2)
        mean_diff = (scores1 - scores2).mean()
        abs_diff = (scores1 - scores2).abs()

        consistency_stats[f"{impl1}_vs_{impl2}"] = {
            "samples": int(mask.sum()),  # Convert to int for JSON serialization
            "correlation": float(
                correlation
            ),  # Convert to float for JSON serialization
            "mean_difference": float(mean_diff),
            "max_difference": float(abs_diff.max()),
            "median_difference": float(abs_diff.median()),
        }

    return consistency_stats


def analyze_image_factors(df):
    """Analyze how image characteristics affect compression efficiency.

    Args:
        df (pd.DataFrame): Benchmark data

    Returns:
        dict: Factor analysis results
    """
    if len(df) < 5:
        return {}

    # Make a copy to avoid modifying the original dataframe
    analysis_data = df.copy()

    # Group by image dimensions (create bins)
    analysis_data["size_category"] = pd.cut(
        analysis_data["width"] * analysis_data["height"],
        bins=[0, 500000, 1000000, 2000000, float("inf")],
        labels=["small", "medium", "large", "very_large"],
    )

    # Analyze compression by size category
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


def analyze_quality_distribution(df):
    """Analyze the distribution of quality scores across formats.

    Args:
        df (pd.DataFrame): Benchmark data

    Returns:
        dict: Quality distribution analysis
    """
    if len(df) == 0 or not df["python_score"].notna().any():
        return {}

    # Make a copy to avoid modifying the original dataframe
    analysis_data = df.copy()

    # Create quality bins (10-point intervals)
    quality_bins = list(range(0, 101, 10))
    analysis_data["quality_bin"] = pd.cut(
        analysis_data["python_score"],
        bins=quality_bins,
        labels=[f"{q}-{q+10}" for q in quality_bins[:-1]],
    )

    # Count samples in each quality bin by format
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
        "mean": float(analysis_data["python_score"].mean()),
        "median": float(analysis_data["python_score"].median()),
        "min": float(analysis_data["python_score"].min()),
        "max": float(analysis_data["python_score"].max()),
    }

    return result
