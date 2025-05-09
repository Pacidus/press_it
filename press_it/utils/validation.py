"""Utilities for input validation."""

import os
import functools
from pathlib import Path


def validate_file_exists(func):
    """Decorator to validate that input file exists before processing."""

    @functools.wraps(func)
    def wrapper(input_path, *args, **kwargs):
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        return func(input_path, *args, **kwargs)

    return wrapper


def validate_quality_range(func):
    """Decorator to validate quality is within valid range (0-100)."""

    @functools.wraps(func)
    def wrapper(input_path, output_path, quality, *args, **kwargs):
        if quality is not None:  # Allow None for formats that don't use quality
            quality = int(quality)  # Ensure quality is an integer
            if quality < 0 or quality > 100:
                raise ValueError(f"Quality must be between 0 and 100, got {quality}")
        return func(input_path, output_path, quality, *args, **kwargs)

    return wrapper


def ensure_output_dir(func):
    """Decorator to ensure output directory exists."""

    @functools.wraps(func)
    def wrapper(input_path, output_path, *args, **kwargs):
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        return func(input_path, output_path, *args, **kwargs)

    return wrapper


def validate_dimensions_format(dimensions):
    """Validate the dimensions string format (e.g., '800x600').

    Args:
        dimensions: Dimensions string to validate

    Returns:
        bool: True if valid, False otherwise
    """
    if not dimensions:
        return False

    parts = dimensions.split("x")
    if len(parts) != 2:
        return False

    width, height = parts

    # Allow empty dimension to maintain aspect ratio
    if width and not width.isdigit():
        return False
    if height and not height.isdigit():
        return False

    return True


def validate_input_files(files):
    """Validate that a list of input files exist.

    Args:
        files: List of file paths to validate

    Returns:
        list: List of existing files

    Raises:
        FileNotFoundError: If any file doesn't exist
    """
    missing = []
    for file_path in files:
        if not os.path.isfile(file_path):
            missing.append(file_path)

    if missing:
        raise FileNotFoundError(f"Files not found: {', '.join(missing)}")

    return files


def validate_ssim_range(func):
    """Decorator to validate SSIM value is within valid range (0-100)."""

    @functools.wraps(func)
    def wrapper(input_path, target_ssim, *args, **kwargs):
        if target_ssim < 0 or target_ssim > 100:
            raise ValueError(
                f"Target SSIM must be between 0 and 100, got {target_ssim}"
            )
        return func(input_path, target_ssim, *args, **kwargs)

    return wrapper


def validate_output_format(func):
    """Decorator to validate the output format is supported."""

    @functools.wraps(func)
    def wrapper(input_path, output_path, *args, **kwargs):
        # Get the extension without the dot
        ext = os.path.splitext(output_path)[1].lower()[1:]

        # List of supported formats
        supported_formats = ["jpg", "jpeg", "png", "webp", "avif"]

        if ext not in supported_formats:
            raise ValueError(f"Unsupported output format: {ext}")

        return func(input_path, output_path, *args, **kwargs)

    return wrapper
