"""Command-line interface for press_it."""

import argparse
import os
import shutil
import subprocess
import tempfile
import sys
from pathlib import Path

from ssimulacra2 import compute_ssimulacra2_with_alpha

from press_it import __version__
from press_it.utils import check_dependencies, resize_image
from press_it.encoders import (
    quality_optimizer,
    encode_webp,
    encode_avif,
    encode_mozjpeg,
    optimize_png,
    FILE_EXTENSIONS,
)


# Required system dependencies
DEPENDENCIES = [
    "pngcrush",
    "avifenc",
    "avifdec",
    "magick",
    "cjpeg",
    "cwebp",
    "dwebp",
]


def main():
    """Main entry point for the press_it CLI."""
    parser = argparse.ArgumentParser(
        description="Optimize images for target SSIM using multiple encoders",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("input_image", help="Path to source image file")
    parser.add_argument("target_ssim", type=float, help="Target SSIM value (0-100)")
    parser.add_argument(
        "--version", "-V", action="version", version=f"press-it {__version__}"
    )
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
        help="Output directory. If not provided, uses current directory",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed progress information during compression",
    )

    args = parser.parse_args()

    # Set SSIM calculation method
    get_ssim = compute_ssimulacra2_with_alpha

    # Verify system dependencies
    try:
        check_dependencies(DEPENDENCIES)
    except Exception as e:
        print(str(e), file=sys.stderr)
        return 1

    # Create optimized versions with different encoders and find the best
    results = {}

    if args.verbose:
        print(f"Processing image: {args.input_image}")
        print(f"Target SSIM quality: {args.target_ssim}")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Convert input to PNG for consistent comparison
        reference_png = os.path.join(temp_dir, "reference.png")
        try:
            if args.verbose:
                print("Converting image to reference format...")
            subprocess.run(
                ["magick", args.input_image, "-alpha", "off", reference_png],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError as e:
            print(f"Error converting input image: {e}", file=sys.stderr)
            return 1

        if args.resize:
            try:
                if args.verbose:
                    print(f"Resizing image to {args.resize}...")
                resize_image(reference_png, args.resize)
            except subprocess.CalledProcessError as e:
                print(f"Error resizing image: {e}", file=sys.stderr)
                return 1

        # Create optimized versions with each encoder
        encoders = {
            "mozjpeg": quality_optimizer(get_ssim)(encode_mozjpeg),
            "webp": quality_optimizer(get_ssim)(encode_webp),
            "avif": quality_optimizer(get_ssim)(encode_avif),
        }

        # Try PNG optimization separately (no quality parameter)
        try:
            base_name = os.path.splitext(os.path.basename(args.input_image))[0]
            png_output = os.path.join(temp_dir, f"{base_name}_100.png")
            if args.verbose:
                print("Optimizing with PNG...")
            subprocess.run(
                ["pngcrush", reference_png, png_output],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            png_size = os.path.getsize(png_output)
            results["png"] = {
                "size": png_size,
                "quality": 100,  # Quality is meaningless for PNG
            }
            print(f"best png: size={png_size}, quality=100")
        except Exception as e:
            print(f"PNG encoding failed: {str(e)}")

        # Process through all other encoders
        for name, encoder in encoders.items():
            try:
                if args.verbose:
                    print(f"Optimizing with {name}...")
                quality, size = encoder(
                    reference_png, temp_dir, args.target_ssim, args.input_image
                )
                print(f"best {name}: size={size}, quality={quality}")
                if quality is not None:
                    results[name] = {
                        "size": size,
                        "quality": quality,
                    }
            except Exception as e:
                print(f"{name} encoding failed: {str(e)}")

        if not results:
            print("No successful encodings achieved", file=sys.stderr)
            return 1

        # Determine best format by size
        best_format = min(results, key=lambda x: results[x]["size"])
        best_result = results[best_format]

        if args.verbose:
            print(
                f"Best format identified: {best_format} with quality {best_result['quality']}"
            )

        # Set output directory
        output_dir = "./"
        if args.output:
            output_dir = args.output
            os.makedirs(output_dir, exist_ok=True)

        # Create output filename
        base_name = os.path.splitext(os.path.basename(args.input_image))[0]
        output_file = (
            f"{base_name}_{best_result['quality']}.{FILE_EXTENSIONS[best_format]}"
        )

        # Move best result to output directory
        source_path = os.path.join(temp_dir, output_file)
        output_path = os.path.join(output_dir, output_file)

        # Handle PNG format which has a different naming convention in our temp dir
        if best_format == "png":
            source_path = os.path.join(temp_dir, f"{base_name}_100.png")

        # Copy the file to the destination (handle already-exists case)
        if os.path.exists(output_path):
            os.remove(output_path)
        try:
            if args.verbose:
                print(f"Saving optimized image to: {output_path}")
            shutil.copy2(source_path, output_path)
        except Exception as e:
            print(f"Error saving output file: {e}", file=sys.stderr)
            return 1

        print(
            "\n"
            "Optimization complete!\n"
            f"    Best format: {best_format} ({best_result['size']} bytes)\n"
            f"    Output file: {output_path}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
