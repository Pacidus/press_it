"""Quality optimization for image compression."""

import os
import functools
from tqdm import tqdm

from press_it.utils.validation import validate_file_exists, validate_ssim_range


def quality_optimizer(get_ssim_func):
    """Decorator to convert encoder function into binary search.

    This decorator transforms a simple encoder function into a function that performs
    a binary search to find the optimal quality parameter that meets or exceeds
    a target SSIM quality while minimizing file size.

    Args:
        get_ssim_func: Function that takes two paths (original and encoded) and
                      returns an SSIM quality score

    Returns:
        A decorator function that performs the binary search
    """

    def decorator(encoder_func):
        @functools.wraps(encoder_func)
        @validate_file_exists
        @validate_ssim_range
        def binary_search(input_path, target_ssim, output_name=None, iterations=7):
            """Find optimal quality setting for the target SSIM.

            Args:
                input_path: Path to input image
                target_ssim: Target SSIM score (0-100)
                output_name: Base name for output file (default: derive from input)
                iterations: Number of binary search iterations

            Returns:
                tuple: (quality, size, output_path)
            """
            # Get format name from encoder function name
            format_name = encoder_func.__name__.split("_")[1]

            # Get file extension for this format
            from press_it.core.encoders import get_extension

            extension = get_extension(format_name)

            # Derive output name if not provided
            if output_name is None:
                base_name = os.path.splitext(os.path.basename(input_path))[0]
            else:
                base_name = output_name

            # Track best result
            best = (None, None, None)  # quality, size, output_path

            # Binary search algorithm
            search_bits = list(range(iterations))[::-1]  # Most significant bit first
            mid = 0  # Quality value

            for i in (pbar := tqdm(search_bits, desc=f"{format_name}: searching")):
                if mid == 100:
                    continue  # Already at max quality

                # Adjust quality value based on current bit
                bit_value = 0b1 << i
                mid = mid + bit_value

                # Clamp to valid range
                if mid > 100:
                    mid = 100

                # Create output filenames
                output_path = f"{base_name}_{mid}.{extension}"

                try:
                    # Encode image with current quality
                    decoded_png = encoder_func(input_path, output_path, mid)

                    # Measure quality
                    current_ssim = get_ssim_func(input_path, decoded_png)

                    # Update progress bar
                    pbar.set_description(
                        f"{format_name}: quality={mid} ssim={current_ssim:.1f}"
                    )

                    # If quality is good enough, save result and continue search at lower quality
                    if current_ssim >= target_ssim:
                        best = (mid, os.path.getsize(output_path), output_path)
                        mid -= bit_value  # Try a lower quality next

                except Exception as e:
                    print(f"Error at quality {mid}: {e}")
                    continue

            return best

        return binary_search

    return decorator


def optimize_all_formats(get_ssim_func, formats=None):
    """Create a function that tests all formats and returns the best result.

    Args:
        get_ssim_func: Function to calculate SSIM
        formats: List of formats to test (default: all)

    Returns:
        Function: A function that takes input_path and target_ssim,
                 and returns the best result across all formats
    """
    # Import here to avoid circular imports
    from press_it.core.encoders import get_encoder

    # Apply quality optimizer to each encoder
    optimized_encoders = {}

    # Default formats to test
    if formats is None:
        formats = ["mozjpeg", "webp", "avif", "png"]

    # Create optimized encoder for each format
    for fmt in formats:
        encoder = get_encoder(fmt)
        optimized_encoders[fmt] = quality_optimizer(get_ssim_func)(encoder)

    @validate_file_exists
    @validate_ssim_range
    def find_best_format(input_path, target_ssim, output_dir=None, base_name=None):
        """Find the best format and quality for the target SSIM.

        Args:
            input_path: Path to input image
            target_ssim: Target SSIM score (0-100)
            output_dir: Directory for output files (default: same as input)
            base_name: Base name for output files (default: derive from input)

        Returns:
            dict: Result with format, quality, size, output_path
        """
        # Set up output directory
        if output_dir is None:
            output_dir = os.path.dirname(input_path)

        # Create output directory if needed
        os.makedirs(output_dir, exist_ok=True)

        # Set up base name
        if base_name is None:
            base_name = os.path.splitext(os.path.basename(input_path))[0]

        # Full path for output base
        output_base = os.path.join(output_dir, base_name)

        # Test each format
        results = {}

        for fmt, optimizer in optimized_encoders.items():
            try:
                quality, size, output_path = optimizer(
                    input_path, target_ssim, output_base
                )

                if quality is not None:
                    results[fmt] = {
                        "format": fmt,
                        "quality": quality,
                        "size": size,
                        "output_path": output_path,
                    }
                    print(f"Best {fmt}: quality={quality}, size={size}")

            except Exception as e:
                print(f"Error optimizing {fmt}: {e}")

        # If no format succeeded, raise error
        if not results:
            raise RuntimeError("Failed to optimize image with any format")

        # Find the best format by size
        best_format = min(results, key=lambda x: results[x]["size"])

        return results[best_format]

    return find_best_format
