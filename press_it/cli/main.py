#!/usr/bin/env python
"""Main entry point for press_it when run as a script."""

import sys
import argparse
from press_it import __version__
from press_it.utils.subprocess_utils import check_dependencies


def create_parent_parser():
    """Create a parent parser with common arguments."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed progress information",
    )

    # Add version information for parent parser epilog
    parser.epilog = f"press_it {__version__}"

    return parser


def run_with_args(parse_func, run_func, dependencies):
    """Run the CLI with parsed arguments and dependency checking.

    Args:
        parse_func: Function to parse arguments
        run_func: Function to run with parsed arguments
        dependencies: List of required system dependencies

    Returns:
        int: Exit code
    """
    # Parse arguments
    args = parse_func()

    # Verify system dependencies
    try:
        check_dependencies(dependencies)
    except Exception as e:
        print(str(e), file=sys.stderr)
        return 1

    # Run with parsed arguments
    return run_func(args)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="press_it image compression and benchmarking tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Set up subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Compression command
    compress_parser = subparsers.add_parser(
        "compress", help="Compress an image to target SSIM quality"
    )
    compress_parser.add_argument("input_image", help="Path to source image file")
    compress_parser.add_argument(
        "target_ssim", type=float, help="Target SSIM value (0-100)"
    )
    compress_parser.add_argument(
        "--resize",
        "-r",
        type=str,
        help=(
            "Resize the image (width)x(height). "
            "If one dimension is not set, the aspect ratio is respected"
        ),
    )
    compress_parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output directory or file. If not provided, uses current directory",
    )
    compress_parser.add_argument(
        "--formats",
        "-f",
        type=str,
        help="Comma-separated list of formats to try (mozjpeg,webp,avif,png)",
    )
    compress_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed progress information",
    )

    # Benchmark command
    benchmark_parser = subparsers.add_parser(
        "benchmark", help="Benchmark compression formats with SSIMULACRA2"
    )
    benchmark_parser.add_argument(
        "--num-images",
        "-n",
        type=int,
        default=0,
        help="Number of images to process (0 for infinite until Ctrl+C)",
    )
    benchmark_parser.add_argument(
        "--output", "-o", type=str, help="Output Parquet file path"
    )
    benchmark_parser.add_argument(
        "--temp-dir",
        "-t",
        type=str,
        help="Directory to store temporary files during benchmark",
    )
    benchmark_parser.add_argument(
        "--analyze", action="store_true", help="Analyze benchmark results after running"
    )
    benchmark_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed progress information",
    )

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze benchmark results")
    analyze_parser.add_argument(
        "benchmark_file",
        nargs="?",
        help="Path to benchmark file (default: most recent)",
    )
    analyze_parser.add_argument(
        "--combine",
        action="store_true",
        help="Combine all benchmark files in current directory",
    )
    analyze_parser.add_argument(
        "--report", type=str, help="Generate a markdown report at the specified path"
    )

    # Version command
    version_parser = subparsers.add_parser("version", help="Show version information")

    # Parse arguments
    args = parser.parse_args()

    # Handle commands
    if args.command == "compress":
        from press_it.cli.compression_cli import run_compression

        return run_compression(args)

    elif args.command == "benchmark":
        from press_it.cli.benchmark_cli import run_benchmark

        return run_benchmark(args)

    elif args.command == "analyze":
        from press_it.cli.benchmark_cli import handle_analysis

        return handle_analysis(args)

    elif args.command == "version":
        from press_it import __version__

        print(f"press_it version {__version__}")
        return 0

    else:
        # No command specified, show help
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
