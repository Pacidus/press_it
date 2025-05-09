"""CLI interface for press_it compression."""

import argparse
import os
import sys
import shutil
from pathlib import Path

from press_it import __version__
from press_it.cli.main import create_parent_parser, run_with_args
from press_it.core.compression import compress_with_target_quality, bulk_compress
from press_it.utils.validation import validate_dimensions_format


# Required system dependencies
DEPENDENCIES = [
    "magick",
    "cjpeg",
    "cwebp",
    "dwebp",
    "avifenc",
    "avifdec",
    "pngcrush",
]


def parse_compression_args():
    """Parse command-line arguments for compression.

    Returns:
        Namespace: Parsed arguments
    """
    parent_parser = create_parent_parser()

    parser = argparse.ArgumentParser(
        description="Optimize images for target SSIM using multiple encoders",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[parent_parser],
        epilog=parent_parser.epilog,
    )

    parser.add_argument("input_image", help="Path to source image file")

    parser.add_argument("target_ssim", type=float, help="Target SSIM value (0-100)")

    parser.add_argument(
        "--resize",
        "-r",
        type=str,
        help=(
            "Resize the image (width)x(height). "
            "If one dimension is not set, the aspect ratio is respected"
        ),
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output directory or file. If not provided, uses current directory",
    )

    parser.add_argument(
        "--formats",
        "-f",
        type=str,
        help="Comma-separated list of formats to try (mozjpeg,webp,avif,png)",
    )

    parser.add_argument(
        "--keep-temp",
        "-k",
        action="store_true",
        help="Keep temporary files after compression",
    )

    return parser.parse_args()


def run_compression(args):
    """Run the compression with provided arguments.

    Args:
        args: Parsed command-line arguments

    Returns:
        int: Exit code (0 for success, non-zero for error)
    """
    # Validate resize dimensions if provided
    if args.resize and not validate_dimensions_format(args.resize):
        print(f"Invalid resize dimensions: {args.resize}", file=sys.stderr)
        print("Format should be WIDTHxHEIGHT (e.g., 800x600)", file=sys.stderr)
        print("To maintain aspect ratio, use WIDTHx or xHEIGHT", file=sys.stderr)
        return 1

    # Parse formats if provided
    formats = None
    if args.formats:
        formats = [f.strip() for f in args.formats.split(",")]

    # Ensure input file exists
    if not os.path.isfile(args.input_image):
        print(f"Input file not found: {args.input_image}", file=sys.stderr)
        return 1

    # Determine output path (directory or specific file)
    output_dir = None
    output_file = None
    if args.output:
        if os.path.isdir(args.output) or args.output.endswith(("/", "\\")):
            # It's a directory
            output_dir = args.output
        else:
            # Specific output file
            output_dir = os.path.dirname(args.output)
            if not output_dir:  # If there's no directory part, use current dir
                output_dir = "."
            output_file = os.path.basename(args.output)

    try:
        print(f"Processing: {args.input_image}")
        print(f"Target SSIM quality: {args.target_ssim}")

        if args.resize:
            print(f"Resizing to: {args.resize}")

        if formats:
            print(f"Testing formats: {', '.join(formats)}")

        # Run compression
        result = compress_with_target_quality(
            args.input_image,
            args.target_ssim,
            formats=formats,
            output_dir=output_dir,
            resize=args.resize,
            keep_temp=args.keep_temp,
        )

        # If a specific output file was requested, rename the result
        if output_file:
            output_path = os.path.join(output_dir, output_file)
            shutil.move(result["output_path"], output_path)
            result["output_path"] = output_path

        print("\nOptimization complete!")
        print(f"  Best format: {result['format']} (quality: {result['quality']})")
        print(f"  Size: {result['size']} bytes")
        print(f"  Output file: {result['output_path']}")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def main():
    """Main entry point for the compression CLI."""
    return run_with_args(parse_compression_args, run_compression, DEPENDENCIES)


if __name__ == "__main__":
    sys.exit(main())
