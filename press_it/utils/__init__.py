"""Utility functions for press_it."""

# Import key functions to make them available at the utils package level
from press_it.utils.subprocess_utils import (
    run_command,
    with_error_handling,
    check_command_exists,
    check_dependencies,
)

from press_it.utils.image import (
    convert_image,
    resize_image,
    ensure_png,
    make_temp_copy,
    get_image_dimensions,
    create_blank_image,
    create_gradient_image,
)

from press_it.utils.validation import (
    validate_file_exists,
    validate_quality_range,
    ensure_output_dir,
    validate_dimensions_format,
    validate_input_files,
    validate_ssim_range,
    validate_output_format,
)

# Define what's available when doing "from press_it.utils import *"
__all__ = [
    # Subprocess utilities
    "run_command",
    "with_error_handling",
    "check_command_exists",
    "check_dependencies",
    # Image utilities
    "convert_image",
    "resize_image",
    "ensure_png",
    "make_temp_copy",
    "get_image_dimensions",
    "create_blank_image",
    "create_gradient_image",
    # Validation utilities
    "validate_file_exists",
    "validate_quality_range",
    "ensure_output_dir",
    "validate_dimensions_format",
    "validate_input_files",
    "validate_ssim_range",
    "validate_output_format",
]
