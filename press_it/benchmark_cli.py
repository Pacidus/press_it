"""Command-line interface for press_it benchmarking."""

import argparse
import sys
import os
from pathlib import Path

from press_it import __version__
from press_it.utils import check_dependencies
from press_it.benchmark.core import BenchmarkRunner
from press_it.benchmark.engines import (
    PYTHON_SSIMULACRA2_VERSION,
    CPP_SSIMULACRA2_VERSION,
    RUST_SSIMULACRA2_VERSION,
)


# Required system dependencies for the benchmark
DEPENDENCIES = [
    "magick",
    "cjpeg",
    "cwebp",
    "dwebp",
    "avifenc",
    "avifdec",
]


def main():
    """Main entry point for the benchmark CLI."""
    parser = argparse.ArgumentParser(
        description="Benchmark different compression formats with SSIMULACRA2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--num-images",
        "-n",
        type=int,
        default=0,
        help="Number of images to process (0 for infinite until Ctrl+C)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="ssimulacra2_benchmark.parquet",
        help="Output Parquet file path",
    )

    parser.add_argument(
        "--temp-dir",
        "-t",
        type=str,
        default=None,
        help="Directory to store temporary files during benchmark (default: auto-generated system temp dir)",
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
        help="Keep temporary image files after benchmark (default: delete)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed progress information during benchmark",
    )

    parser.add_argument(
        "--version",
        "-V",
        action="version",
        version=(
            f"press-benchmark {__version__}\n"
            f"Python SSIMULACRA2: {PYTHON_SSIMULACRA2_VERSION or 'Not available'}\n"
            f"C++ SSIMULACRA2: {CPP_SSIMULACRA2_VERSION or 'Not available'}\n"
            f"Rust SSIMULACRA2: {RUST_SSIMULACRA2_VERSION or 'Not available'}"
        ),
    )

    args = parser.parse_args()

    # Verify system dependencies
    try:
        check_dependencies(DEPENDENCIES)
    except Exception as e:
        print(str(e), file=sys.stderr)
        return 1

    # Run the benchmark
    try:
        print(f"Starting benchmark with {args.num_images or 'infinite'} images...")
        benchmark = BenchmarkRunner(
            output_file=args.output,
            temp_dir=args.temp_dir,
            quality_min=args.quality_min,
            quality_max=args.quality_max,
            verbose=args.verbose,
            keep_images=args.keep_images,
        )

        num_processed = benchmark.run(num_images=args.num_images)

        print(f"\nBenchmark completed successfully! Processed {num_processed} images.")
        print(f"Results saved to: {benchmark.output_file}")
        return 0

    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.")
        return 0

    except Exception as e:
        print(f"Error during benchmark: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
