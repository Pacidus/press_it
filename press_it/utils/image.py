"""Utilities for image manipulation."""

import os
import functools
import shutil
import tempfile
from pathlib import Path

from press_it.utils.subprocess_utils import run_command, with_error_handling


@with_error_handling
def convert_image(input_path, output_path, options=None):
    """Convert an image using ImageMagick.

    Args:
        input_path: Path to input image
        output_path: Path for output image
        options: Additional ImageMagick options as list

    Returns:
        Path to output image
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cmd = ["magick", str(input_path)]

    if options:
        cmd.extend(options)

    cmd.append(str(output_path))

    run_command(cmd, check=True)

    return output_path


@with_error_handling
def resize_image(image_path, dimensions, in_place=True):
    """Resize an image to the specified dimensions.

    Args:
        image_path: Path to the image
        dimensions: Dimensions in WxH format (e.g., "800x600")
        in_place: Whether to resize in place or return a new image

    Returns:
        Path to the resized image
    """
    output_path = (
        image_path
        if in_place
        else f"{os.path.splitext(image_path)[0]}_resized{os.path.splitext(image_path)[1]}"
    )

    return convert_image(image_path, output_path, ["-resize", dimensions])


@with_error_handling
def ensure_png(input_path, output_path=None, remove_alpha=True):
    """Ensure an image is in PNG format.

    Args:
        input_path: Path to input image
        output_path: Path for output PNG (optional)
        remove_alpha: Whether to remove alpha channel

    Returns:
        Path to the PNG image
    """
    if output_path is None:
        output_path = f"{os.path.splitext(input_path)[0]}.png"

    options = []
    if remove_alpha:
        options.extend(["-alpha", "off"])

    return convert_image(input_path, output_path, options)


def make_temp_copy(func):
    """Decorator to create a temporary copy of an image before processing.

    This preserves the original image while allowing modifications.
    """

    @functools.wraps(func)
    def wrapper(input_path, *args, **kwargs):
        # Create a temporary copy with a unique name
        tmp_dir = tempfile.gettempdir()
        basename = os.path.basename(input_path)
        tmp_path = os.path.join(tmp_dir, f"temp_{basename}")

        # Copy the file
        shutil.copy2(input_path, tmp_path)

        try:
            # Call the original function with the temp file
            result = func(tmp_path, *args, **kwargs)
            return result
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    return wrapper


def get_image_dimensions(image_path):
    """Get the dimensions of an image.

    Args:
        image_path: Path to the image

    Returns:
        tuple: (width, height)
    """
    from PIL import Image

    with Image.open(image_path) as img:
        return img.size


def create_blank_image(output_path, width, height, color="white"):
    """Create a blank image with specified dimensions and color.

    Args:
        output_path: Where to save the image
        width: Image width in pixels
        height: Image height in pixels
        color: Background color (default: white)

    Returns:
        Path to created image
    """
    from PIL import Image

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create image and save
    img = Image.new("RGB", (width, height), color=color)
    img.save(output_path)

    return output_path


def create_gradient_image(output_path, width, height):
    """Create a test gradient image.

    Args:
        output_path: Where to save the image
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        Path to created image
    """
    from PIL import Image

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create a new image with RGB mode
    img = Image.new("RGB", (width, height))

    # Fill with a gradient
    for x in range(width):
        for y in range(height):
            r = int(255 * x / width)
            g = int(255 * y / height)
            b = int(255 * (x + y) / (width + height))
            img.putpixel((x, y), (r, g, b))

    # Save the image
    img.save(output_path)

    return output_path
