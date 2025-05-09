#!/usr/bin/env python3
"""Script to analyze press_it benchmark data."""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Try to import matplotlib, but don't fail if not available
try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Plotting features will be disabled.")

from press_it.benchmark.analysis import (
    load_benchmark_data,
    get_benchmark_files,
    combine_benchmark_files,
    summarize_benchmark_data,
    print_summary,
    analyze_quality_vs_size,
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

    # Plot each compression type with a different color
    for comp_type, data in analysis_df.groupby("compression_type"):
        plt.plot(
            data["best_score"],
            data["compression_ratio"],
            "o-",
            label=comp_type,
            alpha=0.7,
        )

    plt.xlabel("Perceptual Quality Score (SSIMULACRA2)")
    plt.ylabel("Compression Ratio")
    plt.title("Quality vs. Compression Ratio by Format")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Add annotations for some key points
    for comp_type, data in analysis_df.groupby("compression_type"):
        # Annotate best compression ratio point
        best_ratio_idx = data["compression_ratio"].idxmax()
        best_ratio_row = data.loc[best_ratio_idx]
        plt.annotate(
            f"{comp_type}: {best_ratio_row['compression_ratio']:.1f}x",
            (best_ratio_row["best_score"], best_ratio_row["compression_ratio"]),
            xytext=(10, 5),
            textcoords="offset points",
            fontsize=8,
        )

    # Save or show the plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    else:
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

    # Generate summary
    print("\nGenerating summary statistics...")
    summary = summarize_benchmark_data(df)
    print_summary(summary)

    # Save summary as JSON
    if args.output:
        import json

        summary_path = output_dir / "benchmark_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary to {summary_path}")

    # Analyze quality vs. size
    print("\nAnalyzing quality vs. size relationship...")
    analysis_output = output_dir / "quality_vs_size.csv" if args.output else None
    analysis_df = analyze_quality_vs_size(df, analysis_output)

    # Generate plots if requested
    if args.plot:
        if MATPLOTLIB_AVAILABLE:
            print("\nGenerating plots...")
            try:
                plot_path = (
                    output_dir / "quality_vs_compression.png" if args.output else None
                )
                plot_quality_vs_size(analysis_df, plot_path)
            except Exception as e:
                print(f"Error generating plots: {e}")
        else:
            print("\nPlotting requested but matplotlib is not available.")
            print("To enable plotting, install matplotlib: pip install matplotlib")

    print("\nAnalysis complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
