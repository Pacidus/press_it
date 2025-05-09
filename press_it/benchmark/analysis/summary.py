"""Summary statistics for benchmark analysis."""

import pandas as pd
from press_it.benchmark.analysis.core import (
    add_categorical_columns,
    get_best_score_column,
)
from press_it.benchmark.analysis.efficiency import analyze_efficiency_by_quality


def summarize_benchmark_data(df):
    """Generate summary statistics from benchmark data.

    Args:
        df: DataFrame with benchmark data

    Returns:
        dict: Summary statistics
    """
    # Calculate compression_ratio if not present
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

    # Add categorical columns
    df = add_categorical_columns(df)

    summary = {
        "total_samples": len(df),
        "compression_types": {},
        "image_dimensions": {},
        "quality_range": {},
        "score_stats": {},
        "compression_stats": {},
        "implementation_availability": {},
    }

    # Handle empty dataframe
    if len(df) == 0:
        return summary

    # Count by compression type
    if "compression_type" in df.columns:
        compression_counts = df["compression_type"].value_counts().to_dict()
        summary["compression_types"] = {
            comp_type: {
                "count": count,
                "percentage": (count / len(df)) * 100,
            }
            for comp_type, count in compression_counts.items()
        }

    # Image dimensions stats
    if "width" in df.columns and "height" in df.columns:
        summary["image_dimensions"] = {
            "width": {
                "min": int(df["width"].min()),
                "max": int(df["width"].max()),
                "mean": round(df["width"].mean(), 2),
                "median": int(df["width"].median()),
            },
            "height": {
                "min": int(df["height"].min()),
                "max": int(df["height"].max()),
                "mean": round(df["height"].mean(), 2),
                "median": int(df["height"].median()),
            },
        }

    # Quality settings range by compression type
    if "compression_type" in df.columns and "quality" in df.columns:
        summary["quality_range"] = {
            comp_type: {
                "min": int(group["quality"].min()),
                "max": int(group["quality"].max()),
                "mean": round(group["quality"].mean(), 2),
            }
            for comp_type, group in df.groupby("compression_type")
        }

    # Get the best available score column
    try:
        score_col = get_best_score_column(df)

        # SSIMULACRA2 score statistics
        if df[score_col].notna().any():
            summary["score_stats"]["overall"] = {
                "min": round(df[score_col].min(), 2),
                "max": round(df[score_col].max(), 2),
                "mean": round(df[score_col].mean(), 2),
                "median": round(df[score_col].median(), 2),
                "std": round(df[score_col].std(), 2),
            }

            # Score stats by compression type
            if "compression_type" in df.columns:
                summary["score_stats"]["by_compression_type"] = {
                    comp_type: {
                        "min": round(group[score_col].min(), 2),
                        "max": round(group[score_col].max(), 2),
                        "mean": round(group[score_col].mean(), 2),
                        "median": round(group[score_col].median(), 2),
                    }
                    for comp_type, group in df.groupby("compression_type")
                    if group[score_col].notna().any()
                }
    except ValueError:
        # No valid score columns found
        pass

    # Compression ratio statistics
    if "compression_ratio" in df.columns:
        summary["compression_stats"]["overall"] = {
            "min": round(df["compression_ratio"].min(), 2),
            "max": round(df["compression_ratio"].max(), 2),
            "mean": round(df["compression_ratio"].mean(), 2),
            "median": round(df["compression_ratio"].median(), 2),
        }

        # Compression stats by format
        if "compression_type" in df.columns:
            summary["compression_stats"]["by_compression_type"] = {
                comp_type: {
                    "min": round(group["compression_ratio"].min(), 2),
                    "max": round(group["compression_ratio"].max(), 2),
                    "mean": round(group["compression_ratio"].mean(), 2),
                    "median": round(group["compression_ratio"].median(), 2),
                }
                for comp_type, group in df.groupby("compression_type")
            }

    # Implementation availability
    for impl in ["python", "cpp", "rust"]:
        col_name = f"{impl}_score"
        if col_name in df.columns:
            summary["implementation_availability"][impl] = round(
                (df[col_name].notna().sum() / len(df)) * 100 if len(df) > 0 else 0, 2
            )

    # Most efficient format at different quality levels
    if "quality_bin" in df.columns and "compression_type" in df.columns:
        summary["most_efficient_format"] = analyze_efficiency_by_quality(df)

    return summary


def print_summary(summary):
    """Print a human-readable summary of benchmark results.

    Args:
        summary: Summary statistics from summarize_benchmark_data
    """
    print("\n===== BENCHMARK SUMMARY =====")
    print(f"Total samples: {summary['total_samples']}")

    # Print compression type breakdown
    if summary["compression_types"]:
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
    if "compression_stats" in summary and "overall" in summary["compression_stats"]:
        print("\n--- Compression Ratio Statistics ---")
        print(
            f"Overall: min={summary['compression_stats']['overall']['min']:.2f}x, "
            f"max={summary['compression_stats']['overall']['max']:.2f}x, "
            f"mean={summary['compression_stats']['overall']['mean']:.2f}x, "
            f"median={summary['compression_stats']['overall']['median']:.2f}x"
        )

        if "by_compression_type" in summary["compression_stats"]:
            print("\nBy compression type:")
            for comp_type, stats in summary["compression_stats"][
                "by_compression_type"
            ].items():
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

    # Print implementation availability
    if summary["implementation_availability"]:
        print("\n--- Implementation Availability ---")
        for impl, percentage in summary["implementation_availability"].items():
            print(f"{impl}: {percentage:.2f}%")
