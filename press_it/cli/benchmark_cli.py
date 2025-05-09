"""CLI interface for press_it benchmarking."""

import argparse
import sys
import os
from pathlib import Path

from press_it import __version__
from press_it.benchmark.engines import (
    PYTHON_SSIMULACRA2_VERSION,
    CPP_SSIMULACRA2_VERSION,
    RUST_SSIMULACRA2_VERSION,
)
from press_it.benchmark.runner import BenchmarkRunner
from press_it.benchmark.analysis import (
    find_benchmark_files,
    load_benchmark_data,
    combine_benchmark_files,
    summarize_benchmark_data,
    print_summary,
    generate_benchmark_report,
)
from press_it.cli.main import create_parent_parser, run_with_args
from press_it.utils.subprocess_utils import check_dependencies


# Required system dependencies for the benchmark
DEPENDENCIES = [
    "magick",
    "cjpeg",
    "cwebp",
    "dwebp",
    "avifenc",
    "avifdec",
]


def parse_benchmark_args():
    """Parse command-line arguments for benchmarking.

    Returns:
        Namespace: Parsed arguments
    """
    parent_parser = create_parent_parser()

    parser = argparse.ArgumentParser(
        description="Benchmark different compression formats with SSIMULACRA2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[parent_parser],
        epilog=parent_parser.epilog,
    )

    parser.add_argument(
        "--num-images",
        "-n",
        type=int,
        default=0,
        help="Number of images to process (0 for infinite until Ctrl+C)",
    )

    parser.add_argument("--output", "-o", type=str, help="Output Parquet file path")

    parser.add_argument(
        "--temp-dir",
        "-t",
        type=str,
        help="Directory to store temporary files during benchmark",
    )

    parser.add_argument(
        "--quality-min",
        type=int,
        default=5,
        help="Minimum quality value to test (5-100)",
    )

    parser.add_argument(
        "--quality-max",
        type=int,
        default=95,
        help="Maximum quality value to test (5-100)",
    )

    parser.add_argument(
        "--keep-images",
        "-k",
        action="store_true",
        help="Keep temporary image files after benchmark",
    )

    # Add analysis options
    parser.add_argument(
        "--analyze", action="store_true", help="Analyze benchmark results after running"
    )

    parser.add_argument(
        "--analyze-only",
        type=str,
        help="Only analyze an existing benchmark file without running new tests",
    )

    parser.add_argument(
        "--combine",
        action="store_true",
        help="Combine and analyze all benchmark files in current directory",
    )

    parser.add_argument(
        "--report", type=str, help="Generate a markdown report at the specified path"
    )

    # Set version string to include SSIMULACRA2 versions
    version_str = (
        f"press-benchmark {__version__}\n"
        f"Python SSIMULACRA2: {PYTHON_SSIMULACRA2_VERSION or 'Not available'}\n"
        f"C++ SSIMULACRA2: {CPP_SSIMULACRA2_VERSION or 'Not available'}\n"
        f"Rust SSIMULACRA2: {RUST_SSIMULACRA2_VERSION or 'Not available'}"
    )
    parser.add_argument("--version", "-V", action="version", version=version_str)

    return parser.parse_args()


def handle_analysis(args, benchmark_file=None):
    """Handle benchmark analysis based on command-line arguments.

    Args:
        args: Parsed command-line arguments
        benchmark_file: Path to benchmark file (if available from a run)

    Returns:
        int: Exit code (0 for success, non-zero for error)
    """
    df = None

    # Determine which file to analyze
    if args.analyze_only:
        # Analyze specific file
        df = load_benchmark_data(args.analyze_only)
        if df is None:
            return 1

    elif args.combine:
        # Combine all benchmark files
        benchmark_files = find_benchmark_files()
        if not benchmark_files:
            print("No benchmark files found in the current directory.")
            return 1

        print(f"Combining {len(benchmark_files)} benchmark files...")
        output_path = "combined_benchmark.parquet"
        df = combine_benchmark_files(benchmark_files, output_path)

    elif benchmark_file:
        # Analyze file from current run
        df = load_benchmark_data(benchmark_file)

    else:
        # Analyze most recent file
        benchmark_files = find_benchmark_files()
        if not benchmark_files:
            print("No benchmark files found in the current directory.")
            return 1

        latest_file = benchmark_files[-1]
        print(f"Analyzing most recent benchmark file: {latest_file}")
        df = load_benchmark_data(latest_file)

    # Generate summary
    if df is not None and len(df) > 0:
        summary = summarize_benchmark_data(df)
        print_summary(summary)

        # Generate report if requested
        if args.report:
            report_path = generate_benchmark_report(df, args.report)
            print(f"\nGenerated benchmark report: {report_path}")

        return 0
    else:
        print("No valid benchmark data to analyze.")
        return 1


def run_benchmark(args):
    """Run the benchmark with provided arguments.

    Args:
        args: Parsed command-line arguments

    Returns:
        int: Exit code (0 for success, non-zero for error)
    """
    # Handle analyze-only mode
    if args.analyze_only or args.combine:
        return handle_analysis(args)

    # Run benchmark
    try:
        benchmark = BenchmarkRunner(
            output_file=args.output,
            temp_dir=args.temp_dir,
            quality_min=args.quality_min,
            quality_max=args.quality_max,
            verbose=args.verbose,
            keep_images=args.keep_images,
            target_count=args.num_images,
        )

        # Run the benchmark
        benchmark.run()

        # Analyze results if requested
        if args.analyze:
            print("\nAnalyzing benchmark results...")
            handle_analysis(args, benchmark.output_file)

        return 0

    except Exception as e:
        print(f"Error during benchmark: {e}", file=sys.stderr)
        return 1


def main():
    """Main entry point for the benchmark CLI."""
    return run_with_args(parse_benchmark_args, run_benchmark, DEPENDENCIES)


if __name__ == "__main__":
    sys.exit(main())
