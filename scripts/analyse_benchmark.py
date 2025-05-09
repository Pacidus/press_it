#!/usr/bin/env python3
"""Script to analyze press_it benchmark data."""

import argparse
import sys
import json
from pathlib import Path

import pandas as pd
import numpy as np

# Try to import matplotlib, but don't fail if not available
try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.colors import LinearSegmentedColormap

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Plotting features will be disabled.")

try:
    import seaborn as sns

    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    print(
        "Warning: seaborn not available. Enhanced plotting features will be disabled."
    )

from press_it.benchmark.analysis import (
    load_benchmark_data,
    get_benchmark_files,
    combine_benchmark_files,
    summarize_benchmark_data,
    print_summary,
    analyze_quality_vs_size,
    analyze_encoder_consistency,
    analyze_image_factors,
    analyze_quality_distribution,
)


def plot_quality_vs_size(analysis_df, output_path=None):
    """Create a plot of quality vs. compression ratio.

    Args:
        analysis_df (pd.DataFrame): Analysis data from analyze_quality_vs_size
        output_path (str, optional): Path to save the plot image
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Cannot create plot.")
        return

    if analysis_df is None or len(analysis_df) == 0:
        print("No data to plot")
        return

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Set a custom color palette if seaborn is available
    if SEABORN_AVAILABLE:
        palette = sns.color_palette(
            "viridis", n_colors=len(analysis_df["compression_type"].unique())
        )
        sns.set_style("whitegrid")
    else:
        palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    # Plot each compression type with a different color
    for i, (comp_type, data) in enumerate(analysis_df.groupby("compression_type")):
        color = palette[i % len(palette)]
        plt.plot(
            data["best_score"],
            data["compression_ratio"],
            "o-",
            label=comp_type,
            alpha=0.7,
            color=color,
            markersize=8,
            linewidth=2,
        )

    plt.xlabel("Perceptual Quality Score (SSIMULACRA2)", fontsize=12)
    plt.ylabel("Compression Ratio", fontsize=12)
    plt.title("Quality vs. Compression Ratio by Format", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)

    # Format y-axis to show compression ratio as "Nx"
    plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}x"))

    # Add horizontal and vertical reference lines
    plt.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)  # No compression line

    # Common quality thresholds
    for quality in [60, 70, 80, 90]:
        plt.axvline(x=quality, color="gray", linestyle=":", alpha=0.3)
        plt.text(
            quality, plt.ylim()[0], f"{quality}", ha="center", va="bottom", alpha=0.7
        )

    # Add annotations for some key points
    for comp_type, data in analysis_df.groupby("compression_type"):
        # Annotate best compression ratio point
        if len(data) > 0:
            best_ratio_idx = data["compression_ratio"].idxmax()
            best_ratio_row = data.loc[best_ratio_idx]
            plt.annotate(
                f"{comp_type}: {best_ratio_row['compression_ratio']:.1f}x",
                (best_ratio_row["best_score"], best_ratio_row["compression_ratio"]),
                xytext=(10, 5),
                textcoords="offset points",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
            )

    # Save or show the plot
    if output_path:
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    else:
        plt.tight_layout()
        plt.show()


def plot_format_comparison(analysis_df, output_path=None):
    """Create a plot comparing formats at different quality levels.

    Args:
        analysis_df (pd.DataFrame): Analysis data
        output_path (str, optional): Path to save the plot image
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Cannot create plot.")
        return

    if analysis_df is None or len(analysis_df) == 0:
        print("No data to plot")
        return

    # Define quality bins
    quality_bins = [0, 60, 70, 80, 90, 100]
    labels = [f"{q1}-{q2}" for q1, q2 in zip(quality_bins[:-1], quality_bins[1:])]

    # Create quality bins and calculate average compression for each format in each bin
    analysis_df["quality_bin"] = pd.cut(
        analysis_df["best_score"], bins=quality_bins, labels=labels
    )

    # Group by quality bin and compression type
    grouped = (
        analysis_df.groupby(["quality_bin", "compression_type"])["compression_ratio"]
        .mean()
        .reset_index()
    )

    # Pivot for easier plotting
    pivot_df = grouped.pivot(
        index="quality_bin", columns="compression_type", values="compression_ratio"
    )

    # Plot
    plt.figure(figsize=(12, 8))

    if SEABORN_AVAILABLE:
        ax = sns.heatmap(
            pivot_df,
            annot=True,
            fmt=".1f",
            cmap="viridis",
            linewidths=0.5,
            cbar_kws={"label": "Compression Ratio (higher is better)"},
        )
        plt.title("Format Comparison by Quality Range", fontsize=14)
        plt.ylabel("Quality Range (SSIMULACRA2 Score)", fontsize=12)

        # Highlight the best format in each quality range
        for i, quality_range in enumerate(pivot_df.index):
            best_format = pivot_df.loc[quality_range].idxmax()
            best_value = pivot_df.loc[quality_range, best_format]
            j = list(pivot_df.columns).index(best_format)
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor="red", lw=2))
    else:
        # Fallback if seaborn not available
        ax = pivot_df.plot(kind="bar", figsize=(12, 8))
        plt.title("Format Comparison by Quality Range")
        plt.ylabel("Compression Ratio (higher is better)")
        plt.grid(axis="y", alpha=0.3)
        plt.legend(title="Format")

        # Format y ticks
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}x"))

    # Save or show the plot
    if output_path:
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved format comparison plot to {output_path}")
    else:
        plt.tight_layout()
        plt.show()


def plot_implementation_comparison(df, output_path=None):
    """Create a plot comparing different SSIMULACRA2 implementations.

    Args:
        df (pd.DataFrame): Benchmark data
        output_path (str, optional): Path to save the plot image
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Cannot create plot.")
        return

    implementations = ["python_score", "cpp_score", "rust_score"]

    # Check which implementations have data
    available_impls = [impl for impl in implementations if df[impl].notna().sum() > 5]

    if len(available_impls) < 2:
        print("Not enough data to compare implementations (need at least 2)")
        return

    # Create comparison plots
    fig, axes = plt.subplots(
        1, len(available_impls) - 1, figsize=(14, 6), squeeze=False
    )
    axes = axes.flatten()

    # Use seaborn for prettier plots if available
    if SEABORN_AVAILABLE:
        sns.set_style("whitegrid")

    plot_idx = 0
    impl_names = {"python_score": "Python", "cpp_score": "C++", "rust_score": "Rust"}

    for i, impl1 in enumerate(available_impls):
        for impl2 in available_impls[i + 1 :]:
            if plot_idx >= len(axes):
                break

            # Get data points where both implementations have results
            mask = df[impl1].notna() & df[impl2].notna()
            x = df.loc[mask, impl1]
            y = df.loc[mask, impl2]

            if len(x) < 5:
                continue

            ax = axes[plot_idx]

            # Plot scatter with regression line
            if SEABORN_AVAILABLE:
                sns.regplot(
                    x=x,
                    y=y,
                    ax=ax,
                    scatter_kws={"alpha": 0.5},
                    line_kws={"color": "red"},
                )
            else:
                ax.scatter(x, y, alpha=0.5)

                # Add regression line
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                ax.plot(x, p(x), "r-")

            # Plot identity line (x=y)
            lims = [
                min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1]),
            ]
            ax.plot(lims, lims, "k--", alpha=0.5, zorder=0)
            ax.set_xlim(lims)
            ax.set_ylim(lims)

            # Add correlation coefficient
            corr = x.corr(y)
            ax.annotate(
                f"r = {corr:.3f}",
                xy=(0.05, 0.95),
                xycoords="axes fraction",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
            )

            mean_diff = (y - x).mean()
            ax.annotate(
                f"Mean diff: {mean_diff:.3f}",
                xy=(0.05, 0.85),
                xycoords="axes fraction",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
            )

            ax.set_xlabel(f"{impl_names.get(impl1, impl1)} Score")
            ax.set_ylabel(f"{impl_names.get(impl2, impl2)} Score")
            ax.set_title(
                f"{impl_names.get(impl1, impl1)} vs {impl_names.get(impl2, impl2)}"
            )

            plot_idx += 1

    # Hide any unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].axis("off")

    plt.suptitle("SSIMULACRA2 Implementation Comparison", fontsize=14)

    # Save or show the plot
    if output_path:
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved implementation comparison plot to {output_path}")
    else:
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
        plt.show()


def plot_image_size_impact(df, output_path=None):
    """Create a plot showing impact of image size on compression efficiency.

    Args:
        df (pd.DataFrame): Benchmark data
        output_path (str, optional): Path to save the plot image
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Cannot create plot.")
        return

    if len(df) < 10:
        print("Not enough data to analyze image size impact")
        return

    # Create image size category
    analysis_df = df.copy()
    analysis_df["image_size"] = analysis_df["width"] * analysis_df["height"]
    analysis_df["size_category"] = pd.cut(
        analysis_df["image_size"],
        bins=[0, 500000, 1000000, 2000000, float("inf")],
        labels=[
            "Small (<0.5MP)",
            "Medium (0.5-1MP)",
            "Large (1-2MP)",
            "Very Large (>2MP)",
        ],
    )

    # Group by size category and compression type
    grouped = (
        analysis_df.groupby(["size_category", "compression_type"])["compression_ratio"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    # Filter to ensure we have enough samples
    min_samples = 3
    grouped = grouped[grouped["count"] >= min_samples]

    # Plot
    plt.figure(figsize=(14, 8))

    if SEABORN_AVAILABLE:
        ax = sns.barplot(
            x="size_category",
            y="mean",
            hue="compression_type",
            data=grouped,
            palette="viridis",
            errorbar=("ci", 95),
        )
        plt.title("Impact of Image Size on Compression Efficiency", fontsize=14)
        plt.xlabel("Image Size Category", fontsize=12)
        plt.ylabel("Average Compression Ratio", fontsize=12)
        plt.legend(title="Format", fontsize=10)

        # Format y-axis to show compression ratio as "Nx"
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}x"))
    else:
        # Fallback if seaborn not available
        pivot = grouped.pivot(
            index="size_category", columns="compression_type", values="mean"
        )
        ax = pivot.plot(kind="bar", figsize=(14, 8), rot=0)
        plt.title("Impact of Image Size on Compression Efficiency")
        plt.xlabel("Image Size Category")
        plt.ylabel("Average Compression Ratio")
        plt.grid(axis="y", alpha=0.3)
        plt.legend(title="Format")

        # Format y ticks
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}x"))

    # Add sample counts as text on the bars if using seaborn
    if SEABORN_AVAILABLE:
        # Add sample count above each bar
        for i, row in enumerate(grouped.itertuples()):
            bar_x = i
            bar_height = row.mean
            ax.text(
                bar_x,
                bar_height + 0.1,
                f"n={row.count}",
                ha="center",
                va="bottom",
                fontsize=8,
                alpha=0.7,
            )

    # Save or show the plot
    if output_path:
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved image size impact plot to {output_path}")
    else:
        plt.tight_layout()
        plt.show()


def plot_quality_distribution(df, output_path=None):
    """Create a plot showing distribution of quality scores across formats.

    Args:
        df (pd.DataFrame): Benchmark data
        output_path (str, optional): Path to save the plot image
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Cannot create plot.")
        return

    if len(df) < 10 or not df["python_score"].notna().any():
        print("Not enough data to analyze quality distribution")
        return

    # Get distribution data
    quality_data = analyze_quality_distribution(df)

    if not quality_data or "quality_distribution" not in quality_data:
        print("No quality distribution data available")
        return

    # Create figure with two subplots: histogram and heatmap
    if SEABORN_AVAILABLE:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Plot 1: Histograms by format
        analysis_df = df.copy()
        formats = sorted(analysis_df["compression_type"].unique())
        palette = sns.color_palette("viridis", n_colors=len(formats))

        # Create histograms
        for i, format_name in enumerate(formats):
            format_data = analysis_df[analysis_df["compression_type"] == format_name]
            if len(format_data) > 0 and format_data["python_score"].notna().any():
                sns.histplot(
                    data=format_data,
                    x="python_score",
                    ax=ax1,
                    label=format_name,
                    color=palette[i],
                    alpha=0.6,
                    kde=True,
                    bins=20,
                )

        ax1.set_title("Quality Score Distribution by Format", fontsize=14)
        ax1.set_xlabel("SSIMULACRA2 Score", fontsize=12)
        ax1.set_ylabel("Count", fontsize=12)
        ax1.legend(title="Format")
        ax1.grid(alpha=0.3)

        # Plot 2: Heatmap of quality percentages
        # Convert distribution data to DataFrame for heatmap
        dist_data = {}
        quality_ranges = []

        for format_name, dist in quality_data["quality_distribution"].items():
            dist_data[format_name] = dist["percentages"]
            if not quality_ranges:
                quality_ranges = list(dist["percentages"].keys())

        if dist_data:
            dist_df = pd.DataFrame(dist_data)

            # Create heatmap
            sns.heatmap(
                dist_df.T,  # Transpose to have formats as rows
                annot=True,
                fmt=".1f",
                cmap="viridis",
                ax=ax2,
                cbar_kws={"label": "Percentage of samples (%)"},
            )

            ax2.set_title("Quality Score Distribution (%) by Format", fontsize=14)
            ax2.set_xlabel("Quality Range", fontsize=12)
            ax2.set_ylabel("Format", fontsize=12)

    else:
        # Fallback if seaborn not available
        fig, ax = plt.subplots(figsize=(12, 8))

        # Create histograms
        analysis_df = df.copy()
        formats = sorted(analysis_df["compression_type"].unique())

        for format_name in formats:
            format_data = analysis_df[analysis_df["compression_type"] == format_name]
            if len(format_data) > 0 and format_data["python_score"].notna().any():
                ax.hist(
                    format_data["python_score"],
                    bins=10,
                    alpha=0.6,
                    label=format_name,
                )

        ax.set_title("Quality Score Distribution by Format")
        ax.set_xlabel("SSIMULACRA2 Score")
        ax.set_ylabel("Count")
        ax.legend(title="Format")
        ax.grid(alpha=0.3)

    # Save or show the plot
    if output_path:
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved quality distribution plot to {output_path}")
    else:
        plt.tight_layout()
        plt.show()


def main():
    """Main entry point for the analysis script."""
    parser = argparse.ArgumentParser(
        description="Analyze press_it benchmark data",
    )

    parser.add_argument(
        "benchmark_file",
        nargs="?",
        type=str,
        help="Path to benchmark Parquet file (if not provided, will use most recent)",
    )

    parser.add_argument(
        "--combine",
        "-c",
        action="store_true",
        help="Combine all benchmark files in the results directory",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output directory for analysis results",
    )

    parser.add_argument(
        "--plot",
        "-p",
        action="store_true",
        help="Generate plots of the results",
    )

    parser.add_argument(
        "--plot-type",
        "-t",
        choices=["all", "quality", "format", "implementation", "size", "distribution"],
        default="all",
        help="Specific plot types to generate (default: all)",
    )

    parser.add_argument(
        "--format-filter",
        "-f",
        type=str,
        help="Filter results to specific compression formats (comma-separated)",
    )

    args = parser.parse_args()

    # Set up output directory
    output_dir = Path(args.output) if args.output else Path("./analysis_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.combine:
        print("Combining all benchmark files...")
        benchmark_files = get_benchmark_files()
        if not benchmark_files:
            print("No benchmark files found in the default location")
            return 1

        print(f"Found {len(benchmark_files)} benchmark files")
        combined_output = output_dir / "combined_benchmark.parquet"
        df = combine_benchmark_files(benchmark_files, combined_output)

    elif args.benchmark_file:
        # Load specified benchmark file
        print(f"Loading benchmark file: {args.benchmark_file}")
        df = load_benchmark_data(args.benchmark_file)

    else:
        # Load most recent benchmark file
        benchmark_files = get_benchmark_files()
        if not benchmark_files:
            print("No benchmark files found in the default location")
            return 1

        latest_file = benchmark_files[-1]
        print(f"Loading most recent benchmark file: {latest_file}")
        df = load_benchmark_data(latest_file)

    if df is None:
        print("Failed to load benchmark data")
        return 1

    # Apply format filter if specified
    if args.format_filter:
        formats = [f.strip() for f in args.format_filter.split(",")]
        if formats:
            original_count = len(df)
            df = df[df["compression_type"].isin(formats)]
            print(
                f"Filtered to {len(df)} rows with formats: {', '.join(formats)} (from {original_count} rows)"
            )

            if len(df) == 0:
                print("No data left after filtering. Check format names.")
                return 1

    # Generate summary
    print("\nGenerating summary statistics...")
    summary = summarize_benchmark_data(df)
    print_summary(summary)

    # Save summary as JSON
    if args.output:
        import json

        summary_path = output_dir / "benchmark_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary to {summary_path}")

    # Analyze implementation consistency
    consistency_stats = analyze_encoder_consistency(df)
    if consistency_stats:
        print("\n--- Implementation Consistency ---")
        for comparison, stats in consistency_stats.items():
            print(
                f"{comparison}: r={stats['correlation']:.3f}, mean diff={stats['mean_difference']:.3f}"
            )

        if args.output:
            consistency_path = output_dir / "implementation_consistency.json"
            with open(consistency_path, "w", encoding="utf-8") as f:
                json.dump(consistency_stats, f, indent=2)
            print(f"Saved consistency analysis to {consistency_path}")

    # Analyze image factors
    image_factor_stats = analyze_image_factors(df)
    if image_factor_stats and "image_size_impact" in image_factor_stats:
        print("\n--- Image Size Impact ---")
        for size, stats in image_factor_stats["image_size_impact"].items():
            print(
                f"{size}: Best format = {stats['best_format']} ({stats['avg_compression']:.2f}x)"
            )

        if args.output:
            factors_path = output_dir / "image_factors.json"
            with open(factors_path, "w", encoding="utf-8") as f:
                json.dump(image_factor_stats, f, indent=2)
            print(f"Saved image factor analysis to {factors_path}")

    # Analyze quality vs. size
    print("\nAnalyzing quality vs. size relationship...")
    analysis_output = output_dir / "quality_vs_size.csv" if args.output else None
    analysis_df = analyze_quality_vs_size(df, analysis_output)

    # Generate plots if requested
    if args.plot and MATPLOTLIB_AVAILABLE:
        print("\nGenerating plots...")
        try:
            # Quality vs compression plot
            if args.plot_type in ["all", "quality"]:
                plot_path = (
                    output_dir / "quality_vs_compression.png" if args.output else None
                )
                plot_quality_vs_size(analysis_df, plot_path)

            # Format comparison plot
            if args.plot_type in ["all", "format"]:
                format_plot_path = (
                    output_dir / "format_comparison.png" if args.output else None
                )
                plot_format_comparison(analysis_df, format_plot_path)

            # Implementation comparison plot
            if args.plot_type in ["all", "implementation"]:
                impl_plot_path = (
                    output_dir / "implementation_comparison.png"
                    if args.output
                    else None
                )
                plot_implementation_comparison(df, impl_plot_path)

            # Image size impact plot
            if args.plot_type in ["all", "size"]:
                size_plot_path = (
                    output_dir / "image_size_impact.png" if args.output else None
                )
                plot_image_size_impact(df, size_plot_path)

            # Quality distribution plot
            if args.plot_type in ["all", "distribution"]:
                dist_plot_path = (
                    output_dir / "quality_distribution.png" if args.output else None
                )
                plot_quality_distribution(df, dist_plot_path)

        except Exception as e:
            print(f"Error generating plots: {e}")
            import traceback

            traceback.print_exc()
    elif args.plot:
        print("\nPlotting requested but matplotlib is not available.")
        print("To enable plotting, install matplotlib: pip install matplotlib")
        if args.plot_type != "all":
            print(f"Requested plot type: {args.plot_type}")

    # Generate a comprehensive report
    if args.output:
        try:
            report_path = output_dir / "analysis_report.md"
            with open(report_path, "w", encoding="utf-8") as f:
                f.write("# SSIMULACRA2 Benchmark Analysis Report\n\n")

                # Add summary stats
                f.write("## Summary\n\n")
                f.write(f"- Total samples: {summary['total_samples']}\n")

                f.write("\n### Compression Type Distribution\n\n")
                for comp_type, stats in summary["compression_types"].items():
                    f.write(
                        f"- {comp_type}: {stats['count']} samples ({stats['percentage']:.2f}%)\n"
                    )

                # Add SSIM stats
                if "score_stats" in summary and "overall" in summary["score_stats"]:
                    f.write("\n### Quality Score Statistics\n\n")
                    f.write(
                        f"- Overall: min={summary['score_stats']['overall']['min']:.2f}, "
                        f"max={summary['score_stats']['overall']['max']:.2f}, "
                        f"mean={summary['score_stats']['overall']['mean']:.2f}\n"
                    )

                # Add compression stats
                f.write("\n### Compression Ratio Statistics\n\n")
                f.write(
                    f"- Overall: min={summary['compression_stats']['overall']['min']:.2f}x, "
                    f"max={summary['compression_stats']['overall']['max']:.2f}x, "
                    f"mean={summary['compression_stats']['overall']['mean']:.2f}x\n\n"
                )

                f.write("By compression type:\n\n")
                for comp_type, stats in summary["compression_stats"][
                    "by_compression_type"
                ].items():
                    f.write(
                        f"- {comp_type}: min={stats['min']:.2f}x, "
                        f"max={stats['max']:.2f}x, "
                        f"mean={stats['mean']:.2f}x\n"
                    )

                # Add most efficient formats
                if (
                    "most_efficient_format" in summary
                    and summary["most_efficient_format"]
                ):
                    f.write("\n## Most Efficient Format by Quality Level\n\n")
                    for quality_range, format_name in summary[
                        "most_efficient_format"
                    ].items():
                        f.write(f"- Quality {quality_range}: **{format_name}**\n")

                # Add implementation consistency
                if consistency_stats:
                    f.write("\n## Implementation Consistency\n\n")
                    for comparison, stats in consistency_stats.items():
                        f.write(f"- {comparison}:\n")
                        f.write(f"  - Correlation: {stats['correlation']:.3f}\n")
                        f.write(
                            f"  - Mean difference: {stats['mean_difference']:.3f}\n"
                        )
                        f.write(f"  - Max difference: {stats['max_difference']:.3f}\n")
                        f.write(f"  - Samples: {stats['samples']}\n\n")

                # Add image size impact
                if image_factor_stats and "image_size_impact" in image_factor_stats:
                    f.write("\n## Image Size Impact\n\n")
                    for size, stats in image_factor_stats["image_size_impact"].items():
                        f.write(
                            f"- {size}: Best format = **{stats['best_format']}** "
                            f"({stats['avg_compression']:.2f}x) based on {stats['samples']} samples\n"
                        )

                # Add plots if generated
                if args.plot and MATPLOTLIB_AVAILABLE:
                    f.write("\n## Visualizations\n\n")

                    if args.plot_type in ["all", "quality"]:
                        f.write("### Quality vs. Compression Ratio\n\n")
                        f.write(
                            "![Quality vs. Compression Ratio](./quality_vs_compression.png)\n\n"
                        )

                    if args.plot_type in ["all", "format"]:
                        f.write("### Format Comparison by Quality Range\n\n")
                        f.write("![Format Comparison](./format_comparison.png)\n\n")

                    if args.plot_type in ["all", "implementation"]:
                        f.write("### Implementation Comparison\n\n")
                        f.write(
                            "![Implementation Comparison](./implementation_comparison.png)\n\n"
                        )

                    if args.plot_type in ["all", "size"]:
                        f.write("### Image Size Impact\n\n")
                        f.write("![Image Size Impact](./image_size_impact.png)\n\n")

                    if args.plot_type in ["all", "distribution"]:
                        f.write("### Quality Distribution\n\n")
                        f.write(
                            "![Quality Distribution](./quality_distribution.png)\n\n"
                        )

                # Add conclusions
                f.write("\n## Conclusions\n\n")

                # Determine best format overall
                best_format = None
                best_ratio = 0

                for comp_type, stats in summary["compression_stats"][
                    "by_compression_type"
                ].items():
                    if stats["mean"] > best_ratio:
                        best_ratio = stats["mean"]
                        best_format = comp_type

                if best_format:
                    f.write(
                        f"- Best overall format: **{best_format}** with average compression ratio of {best_ratio:.2f}x\n"
                    )

                # Add quality level recommendations
                if (
                    "most_efficient_format" in summary
                    and summary["most_efficient_format"]
                ):
                    f.write("- Recommended formats by quality level:\n")
                    for quality_range, format_name in summary[
                        "most_efficient_format"
                    ].items():
                        f.write(f"  - Quality {quality_range}: **{format_name}**\n")

                f.write("\n*Analysis generated by press_it benchmark analysis tool*\n")

            print(f"Generated analysis report at {report_path}")
        except Exception as e:
            print(f"Error generating report: {e}")

    print("\nAnalysis complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
