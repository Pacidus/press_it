"""Image handling for press_it benchmarks."""

import os
import random
import tempfile
from pathlib import Path

import requests
from PIL import Image

from press_it.utils.image import (
    convert_image,
    create_blank_image,
    create_gradient_image,
)
from press_it.utils.subprocess_utils import run_command, with_error_handling


def download_image(output_path, width=None, height=None, grayscale=False, timeout=10):
    """Download a random image from Picsum Photos.

    Args:
        output_path: Where to save the downloaded image
        width: Desired image width (default: random)
        height: Desired image height (default: random)
        grayscale: Whether to download grayscale image
        timeout: Download timeout in seconds

    Returns:
        str: Path to the downloaded image
    """
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Use random dimensions if not specified
    if width is None:
        width = random.choice([800, 1024, 1280, 1600])
    if height is None:
        height = random.choice([600, 768, 960, 1200])

    # Build URL
    image_url = f"https://picsum.photos/{width}/{height}"
    if grayscale:
        image_url += "?grayscale"

    # Download image
    try:
        response = requests.get(image_url, timeout=timeout)
        response.raise_for_status()

        # Create a temp file for the downloaded image
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            temp_jpg = temp_file.name
            temp_file.write(response.content)

        # Convert to PNG (for consistency)
        convert_image(temp_jpg, output_path, ["-alpha", "off"])

        # Remove temporary file
        os.remove(temp_jpg)

        return output_path

    except Exception as e:
        print(f"Error downloading image: {e}")
        # Create a fallback image
        return create_gradient_image(output_path, width, height)


def get_random_image(output_dir, image_id=None, verbose=False):
    """Download a random image or create one if download fails.

    Args:
        output_dir: Directory to save the image
        image_id: Optional ID for naming
        verbose: Whether to show progress information

    Returns:
        str: Path to the image
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate image ID if not provided
    if image_id is None:
        image_id = random.randint(0, 9999)

    image_id_str = f"image_{image_id:04d}"
    output_path = os.path.join(output_dir, f"{image_id_str}.png")

    # Skip download if file already exists
    if os.path.exists(output_path):
        if verbose:
            print(f"Using existing image: {output_path}")
        return output_path

    if verbose:
        print(f"Downloading new image...")

    # Try to download a random image
    try:
        download_image(output_path)

        if verbose:
            img = Image.open(output_path)
            print(f"Downloaded image: {output_path} ({img.width}x{img.height})")

        return output_path

    except Exception as e:
        print(f"Failed to download image: {e}")

        # Try to find a fallback image
        fallback_images = list(Path(output_dir).glob("*.png"))
        if fallback_images:
            fallback_path = str(random.choice(fallback_images))
            print(f"Using fallback image: {fallback_path}")
            return fallback_path

        # If no fallback, create a test image
        print("Creating a test image")
        create_gradient_image(output_path, 800, 600)
        return output_path


@with_error_handling
def encode_with_format(input_png, output_dir, decoded_dir, format_name, quality=None):
    """Encode an image with a specific format and quality.

    Args:
        input_png: Path to input PNG
        output_dir: Directory for compressed output
        decoded_dir: Directory for decoded output
        format_name: Format name (mozjpeg, webp, avif)
        quality: Compression quality (default: random 5-95)

    Returns:
        tuple: (compressed_path, decoded_path, format_name, quality)
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(decoded_dir, exist_ok=True)

    # Get base name without extension
    base_name = os.path.splitext(os.path.basename(input_png))[0]

    # If quality is not provided, select randomly
    if quality is None:
        quality = random.randint(5, 95)

    # Import encoder function from core
    from press_it.core.encoders import get_encoder, get_extension

    # Get encoder and extension
    encoder = get_encoder(format_name)
    extension = get_extension(format_name)

    # Create output paths
    output_path = os.path.join(
        output_dir, f"{base_name}_{format_name}_{quality}.{extension}"
    )

    # Skip if file already exists
    if os.path.exists(output_path):
        decoded_path = os.path.join(
            decoded_dir, f"{base_name}_{format_name}_{quality}_decoded.png"
        )
        if os.path.exists(decoded_path):
            return output_path, decoded_path, format_name, quality

    # Encode image
    decoded_path = encoder(input_png, output_path, quality)

    # Rename decoded path to follow naming convention
    new_decoded_path = os.path.join(
        decoded_dir, f"{base_name}_{format_name}_{quality}_decoded.png"
    )
    if os.path.abspath(decoded_path) != os.path.abspath(new_decoded_path):
        os.rename(decoded_path, new_decoded_path)
        decoded_path = new_decoded_path

    print(f"Compressed image: {output_path} ({format_name}, quality={quality})")
    return output_path, decoded_path, format_name, quality


def encode_random_format(input_png, output_dir, decoded_dir, quality=None):
    """Encode an image with a randomly selected format.

    Args:
        input_png: Path to input PNG
        output_dir: Directory for compressed output
        decoded_dir: Directory for decoded output
        quality: Compression quality (default: random 5-95)

    Returns:
        tuple: (compressed_path, decoded_path, format_name, quality)
    """
    # Choose a random format
    formats = ["mozjpeg", "webp", "avif"]
    format_name = random.choice(formats)

    # Encode with selected format
    return encode_with_format(input_png, output_dir, decoded_dir, format_name, quality)
