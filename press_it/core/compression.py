"""Main compression functionality for press_it."""

import os
import shutil
import tempfile
from pathlib import Path

from press_it.utils.image import ensure_png, resize_image
from press_it.utils.validation import validate_file_exists, validate_ssim_range
from press_it.core.quality import optimize_all_formats


def initialize_compression(input_path, temp_dir=None, resize_dimensions=None):
    """Prepare image for compression by normalizing format.

    Args:
        input_path: Path to input image
        temp_dir: Directory for temporary files (default: system temp)
        resize_dimensions: Optional dimensions to resize (WxH format)

    Returns:
        Path to prepared reference PNG
    """
    # Create temporary directory if needed
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="press_it_")
    else:
        os.makedirs(temp_dir, exist_ok=True)

    # Create reference PNG
    reference_png = os.path.join(temp_dir, "reference.png")

    # Convert to PNG and remove alpha channel
    ensure_png(input_path, reference_png, remove_alpha=True)

    # Resize if needed
    if resize_dimensions:
        resize_image(reference_png, resize_dimensions, in_place=True)

    return reference_png


@validate_file_exists
@validate_ssim_range
def compress_with_target_quality(
    input_path, target_ssim, formats=None, output_dir=None, resize=None, keep_temp=False
):
    """Compress image to target SSIM using optimal format.

    Args:
        input_path: Path to input image
        target_ssim: Target SSIM score (0-100)
        formats: List of formats to try (default: all)
        output_dir: Directory for output file (default: same as input)
        resize: Optional dimensions to resize (WxH format)
        keep_temp: Whether to keep temporary files

    Returns:
        dict: Compression result with format, quality, size, output_path
    """
    # Import SSIMULACRA2
    from ssimulacra2 import compute_ssimulacra2_with_alpha

    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Prepare image
        reference_png = initialize_compression(input_path, temp_dir, resize)

        # Create optimizer for all formats
        optimize = optimize_all_formats(compute_ssimulacra2_with_alpha, formats)

        # Get base name and create a path for optimizer
        base_name = os.path.splitext(os.path.basename(input_path))[0]

        # Find best format within the temporary directory
        result = optimize(
            reference_png,
            target_ssim,
            temp_dir,  # Use temp_dir for all intermediate files
            base_name,
        )

        # Determine the final output path
        if output_dir:
            # Make sure the output directory exists
            os.makedirs(output_dir, exist_ok=True)

            # Create the final output path with proper extension
            format_ext = os.path.splitext(result["output_path"])[1]
            final_path = os.path.join(output_dir, f"{base_name}{format_ext}")

            # Copy just the final optimized file to the output directory
            shutil.copy2(result["output_path"], final_path)

            # Update the result with the new path
            result["output_path"] = final_path
        else:
            # If no output_dir specified, copy to the same directory as input
            input_dir = os.path.dirname(os.path.abspath(input_path))
            format_ext = os.path.splitext(result["output_path"])[1]
            final_path = os.path.join(input_dir, f"{base_name}_optimized{format_ext}")

            # Copy the final result
            shutil.copy2(result["output_path"], final_path)

            # Update the result with the new path
            result["output_path"] = final_path

        # Keep temporary files if requested
        if keep_temp:
            print(f"Temporary files are kept at: {temp_dir}")
            # Prevent deletion of the temp directory
            tempfile._finalizer._exit_files.remove(os.path.abspath(temp_dir))

        return result


def bulk_compress(input_files, target_ssim, output_dir=None, formats=None, resize=None):
    """Compress multiple images with the same settings.

    Args:
        input_files: List of input image paths
        target_ssim: Target SSIM score
        output_dir: Directory for output files
        formats: List of formats to try
        resize: Optional dimensions to resize

    Returns:
        list: Compression results for each input file
    """
    results = []

    for input_path in input_files:
        try:
            print(f"Processing: {input_path}")
            result = compress_with_target_quality(
                input_path,
                target_ssim,
                formats=formats,
                output_dir=output_dir,
                resize=resize,
            )
            results.append({"input_path": input_path, "result": result})
            print(
                f"Compressed to {result['format']} (quality: {result['quality']}, size: {result['size']} bytes)"
            )

        except Exception as e:
            print(f"Error processing {input_path}: {e}")
            results.append({"input_path": input_path, "error": str(e)})

    return results
