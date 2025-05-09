"""Encoders for various image formats."""

import os
import functools
from pathlib import Path

from press_it.utils.subprocess_utils import run_command
from press_it.utils.validation import (
    validate_file_exists,
    validate_quality_range,
    ensure_output_dir,
)

# File extensions for each encoder format
FORMAT_EXTENSIONS = {
    "mozjpeg": "jpg",
    "webp": "webp",
    "avif": "avif",
    "png": "png",
}


# Base decorator composition for all encoders
def encoder_base(func):
    """Apply base decorators to all encoder functions."""
    return validate_file_exists(validate_quality_range(ensure_output_dir(func)))


@encoder_base
def encode_mozjpeg(input_path, output_path, quality):
    """Encode image to JPEG using MozJPEG.

    Args:
        input_path: Path to input PNG image
        output_path: Path for output JPEG image
        quality: Compression quality (1-100)

    Returns:
        Path to decoded PNG (for quality comparison)
    """
    # Run cjpeg to encode the image
    run_command(
        [
            "cjpeg",
            "-quality",
            str(quality),
            "-outfile",
            str(output_path),
            str(input_path),
        ],
        check=True,
    )

    # Decode back to PNG for quality measurement
    decoded_path = f"{os.path.splitext(output_path)[0]}_decoded.png"
    run_command(["magick", str(output_path), str(decoded_path)], check=True)

    return decoded_path


@encoder_base
def encode_webp(input_path, output_path, quality):
    """Encode image to WebP format using cwebp.

    Args:
        input_path: Path to input PNG image
        output_path: Path for output WebP image
        quality: Compression quality (1-100)

    Returns:
        Path to decoded PNG (for quality comparison)
    """
    # Run cwebp to encode the image
    run_command(
        [
            "cwebp",
            "-m",
            "6",
            "-q",
            str(quality),
            str(input_path),
            "-o",
            str(output_path),
        ],
        check=True,
    )

    # Decode back to PNG for quality measurement
    decoded_path = f"{os.path.splitext(output_path)[0]}_decoded.png"
    run_command(["dwebp", str(output_path), "-o", str(decoded_path)], check=True)

    return decoded_path


@encoder_base
def encode_avif(input_path, output_path, quality):
    """Encode image to AVIF format using avifenc.

    Args:
        input_path: Path to input PNG image
        output_path: Path for output AVIF image
        quality: Compression quality (1-100)

    Returns:
        Path to decoded PNG (for quality comparison)
    """
    # Run avifenc to encode the image
    run_command(
        [
            "avifenc",
            "-q",
            str(quality),
            "-j",
            "all",
            "-s",
            "0",
            str(input_path),
            str(output_path),
        ],
        check=True,
    )

    # Decode back to PNG for quality measurement
    decoded_path = f"{os.path.splitext(output_path)[0]}_decoded.png"
    run_command(["avifdec", str(output_path), str(decoded_path)], check=True)

    return decoded_path


@encoder_base
def encode_png(input_path, output_path, quality=None):
    """Optimize PNG using pngcrush.

    Args:
        input_path: Path to input PNG image
        output_path: Path for output optimized PNG
        quality: Ignored for PNG

    Returns:
        Path to output PNG (same as input since no decoding needed)
    """
    # Run pngcrush to optimize the image
    run_command(["pngcrush", str(input_path), str(output_path)], check=True)

    # For PNG, the decoded image is the same as the output
    return output_path


def get_encoder(format_name):
    """Get an encoder function by format name.

    Args:
        format_name: Name of the format (mozjpeg, webp, avif, png)

    Returns:
        Function: The encoder function

    Raises:
        ValueError: If the format is not supported
    """
    encoders = {
        "mozjpeg": encode_mozjpeg,
        "webp": encode_webp,
        "avif": encode_avif,
        "png": encode_png,
    }

    if format_name not in encoders:
        raise ValueError(f"Unsupported format: {format_name}")

    return encoders[format_name]


def get_extension(format_name):
    """Get file extension for a format.

    Args:
        format_name: Name of the format

    Returns:
        str: File extension (without dot)

    Raises:
        ValueError: If the format is not supported
    """
    if format_name not in FORMAT_EXTENSIONS:
        raise ValueError(f"Unsupported format: {format_name}")

    return FORMAT_EXTENSIONS[format_name]
