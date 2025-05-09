"""Main benchmark functionality integrating all components."""

import os
import sys
from pathlib import Path
import pandas as pd

from press_it.benchmark.runner import BenchmarkRunner
from press_it.benchmark.analysis.core import (
    find_benchmark_files,
    load_benchmark_data,
    combine_benchmark_files,
)
from press_it.benchmark.analysis.summary import summarize_benchmark_data, print_summary
from press_it.benchmark.analysis.reporting import (
    generate_benchmark_report,
    generate_json_report,
)


def run_benchmark(
    num_images=0,
    output_file=None,
    temp_dir=None,
    quality_min=5,
    quality_max=95,
    verbose=False,
    keep_images=False,
):
    """Run a benchmark with specified settings.

    Args:
        num_images: Number of images to process (0 for infinite)
        output_file: Path to save results
        temp_dir: Directory for temporary files
        quality_min: Minimum quality to test
        quality_max: Maximum quality to test
        verbose: Whether to show detailed progress
        keep_images: Whether to keep temporary images

    Returns:
        str: Path to output file
    """
    # Create runner
    runner = BenchmarkRunner(
        output_file=output_file,
        temp_dir=temp_dir,
        quality_min=quality_min,
        quality_max=quality_max,
        verbose=verbose,
        keep_images=keep_images,
        target_count=num_images,
    )

    # Run benchmark
    runner.run()

    # Return output file path
    return runner.output_file


def analyze_benchmark(
    file_path=None, output_dir=None, combine=False, report_format="markdown"
):
    """Analyze benchmark results.

    Args:
        file_path: Path to benchmark file (latest if None)
        output_dir: Directory for output files
        combine: Whether to combine multiple benchmark files
        report_format: Format for the report (markdown or json)

    Returns:
        dict: Summary data
    """
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Determine which file to analyze
    if combine:
        # Combine all benchmark files
        files = find_benchmark_files()
        if not files:
            print("No benchmark files found")
            return None

        print(f"Combining {len(files)} benchmark files...")
        combined_path = (
            os.path.join(output_dir, "combined_benchmark.parquet")
            if output_dir
            else "combined_benchmark.parquet"
        )
        df = combine_benchmark_files(files, combined_path)

    elif file_path:
        # Load specified file
        df = load_benchmark_data(file_path)

    else:
        # Find latest file
        files = find_benchmark_files()
        if not files:
            print("No benchmark files found")
            return None

        print(f"Using most recent benchmark file: {files[-1]}")
        df = load_benchmark_data(files[-1])

    if df is None or len(df) == 0:
        print("No valid benchmark data to analyze")
        return None

    # Generate summary
    summary = summarize_benchmark_data(df)
    print_summary(summary)

    # Generate report if output directory specified
    if output_dir:
        if report_format == "markdown":
            report_path = os.path.join(output_dir, "benchmark_report.md")
            generate_benchmark_report(df, report_path)
            print(f"Generated markdown report: {report_path}")

        elif report_format == "json":
            report_path = os.path.join(output_dir, "benchmark_report.json")
            generate_json_report(df, report_path)
            print(f"Generated JSON report: {report_path}")

        # Generate visualizations if matplotlib is available
        try:
            from press_it.benchmark.analysis.visualization import (
                check_visualization_available,
                create_dashboard,
            )

            if check_visualization_available():
                print("Generating visualizations...")
                viz_dir = os.path.join(output_dir, "visualizations")
                plots = create_dashboard(df, viz_dir)
                print(f"Generated {len(plots)} visualization plots in {viz_dir}")

        except ImportError:
            print("Visualizations skipped (matplotlib not available)")

    return summary


def interactive_benchmark():
    """Run an interactive benchmark session."""
    print("Press-it Interactive Benchmark")
    print("==============================")
    print()

    # Ask for benchmark parameters
    try:
        num_images = int(input("Number of images to process (0 for infinite): ") or "0")
        output_file = input("Output file path (default: auto-generated): ") or None
        quality_min = int(input("Minimum quality (5-100, default: 5): ") or "5")
        quality_max = int(input("Maximum quality (5-100, default: 95): ") or "95")
        verbose = input("Verbose output? (y/n, default: n): ").lower() == "y"
        keep_images = input("Keep temporary images? (y/n, default: n): ").lower() == "y"
        analyze = (
            input("Analyze results after benchmark? (y/n, default: y): ").lower() != "n"
        )

        # Run benchmark
        print("\nStarting benchmark...\n")
        output_path = run_benchmark(
            num_images=num_images,
            output_file=output_file,
            quality_min=quality_min,
            quality_max=quality_max,
            verbose=verbose,
            keep_images=keep_images,
        )

        # Analyze if requested
        if analyze:
            print("\nAnalyzing results...\n")
            analyze_benchmark(output_path)

    except KeyboardInterrupt:
        print("\nBenchmark session interrupted.")
    except Exception as e:
        print(f"\nError: {e}")

    print("\nBenchmark session complete.")


if __name__ == "__main__":
    interactive_benchmark()
