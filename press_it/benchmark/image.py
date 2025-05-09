"""Image handling for press_it benchmarks."""

import os
import random
import subprocess
import tempfile
from pathlib import Path

import requests
from PIL import Image


def get_random_image(output_dir, image_id=None, verbose=False):
    """Download a random image from Picsum Photos with random dimensions.

    Args:
        output_dir (Path): Directory to save the downloaded image
        image_id (int, optional): Image identifier for naming
        verbose (bool): Whether to show detailed progress information

    Returns:
        str: Path to the downloaded or fallback image
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_id_str = (
        f"image_{image_id:04d}"
        if image_id is not None
        else f"image_{random.randint(0, 9999):04d}"
    )
    output_path = output_dir / f"{image_id_str}.png"

    # Skip download if file already exists (from a previous run)
    if output_path.exists():
        if verbose:
            print(f"Using existing image: {output_path}")
        return str(output_path)

    try:
        # Get a random image with random dimensions
        width = random.choice([800, 1024, 1280, 1600])
        height = random.choice([600, 768, 960, 1200])

        if verbose:
            print(f"Downloading new image ({width}x{height})...")

        # Simple URL for random image with specified dimensions
        image_url = f"https://picsum.photos/{width}/{height}"

        response = requests.get(image_url, timeout=10)
        response.raise_for_status()

        # Create a temp file for the downloaded image
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_jpg_file:
            temp_jpg = temp_jpg_file.name
            temp_jpg_file.write(response.content)

        if verbose:
            print(f"Converting image to PNG format...")

        # Convert to PNG with alpha removed for consistent comparison
        subprocess.run(
            ["magick", temp_jpg, "-alpha", "off", str(output_path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Remove temporary file
        os.remove(temp_jpg)

        print(f"Downloaded new image: {output_path} ({width}x{height})")
        return str(output_path)

    except Exception as e:
        print(f"Error downloading image: {e}")

        # If download fails, use a fallback local image if available
        fallback_images = list(Path(output_dir).glob("*.png"))
        if fallback_images:
            fallback_path = random.choice(fallback_images)
            print(f"Using fallback image: {fallback_path}")
            return str(fallback_path)

        # If no fallback available, create a simple test image
        try:
            print("Creating a test image instead")
            test_image = Image.new("RGB", (800, 600), color=(73, 109, 137))
            test_image.save(output_path)
            return str(output_path)
        except Exception as e:
            print(f"Failed to create test image: {e}")
            raise


def encode_mozjpeg(input_png, compressed_dir, decoded_dir, quality=None):
    """Encode image to JPEG using MozJPEG.

    Args:
        input_png (str): Path to input PNG image
        compressed_dir (Path): Directory for compressed output
        decoded_dir (Path): Directory for decoded output
        quality (int, optional): Quality setting (default: random 5-95)

    Returns:
        tuple: (compressed_path, decoded_path, encoder_type, quality)
    """
    # Create output directories if they don't exist
    compressed_dir = Path(compressed_dir)
    decoded_dir = Path(decoded_dir)
    compressed_dir.mkdir(parents=True, exist_ok=True)
    decoded_dir.mkdir(parents=True, exist_ok=True)

    img_path = Path(input_png)
    image_id = img_path.stem

    # If quality is not provided, select randomly
    if quality is None:
        quality = random.randint(5, 95)

    output_path = Path(compressed_dir) / f"{image_id}_mozjpeg_{quality}.jpg"
    decoded_png = Path(decoded_dir) / f"{image_id}_mozjpeg_{quality}_decoded.png"

    # Skip compression if file already exists (from a previous run)
    if output_path.exists() and decoded_png.exists():
        return str(output_path), str(decoded_png), "mozjpeg", quality

    try:
        # Compress using MozJPEG
        subprocess.run(
            [
                "cjpeg",
                "-quality",
                str(quality),
                "-outfile",
                str(output_path),
                input_png,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Decode back to PNG for consistent comparison
        subprocess.run(
            ["magick", str(output_path), str(decoded_png)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        print(f"Compressed image: {output_path} (mozjpeg, quality={quality})")
        return str(output_path), str(decoded_png), "mozjpeg", quality

    except Exception as e:
        print(f"Error compressing image with MozJPEG: {e}")
        raise


def encode_webp(input_png, compressed_dir, decoded_dir, quality=None):
    """Encode image to WebP format using cwebp.

    Args:
        input_png (str): Path to input PNG image
        compressed_dir (Path): Directory for compressed output
        decoded_dir (Path): Directory for decoded output
        quality (int, optional): Quality setting (default: random 5-95)

    Returns:
        tuple: (compressed_path, decoded_path, encoder_type, quality)
    """
    # Create output directories if they don't exist
    compressed_dir = Path(compressed_dir)
    decoded_dir = Path(decoded_dir)
    compressed_dir.mkdir(parents=True, exist_ok=True)
    decoded_dir.mkdir(parents=True, exist_ok=True)

    img_path = Path(input_png)
    image_id = img_path.stem

    # If quality is not provided, select randomly
    if quality is None:
        quality = random.randint(5, 95)

    output_path = Path(compressed_dir) / f"{image_id}_webp_{quality}.webp"
    decoded_png = Path(decoded_dir) / f"{image_id}_webp_{quality}_decoded.png"

    # Skip compression if file already exists (from a previous run)
    if output_path.exists() and decoded_png.exists():
        return str(output_path), str(decoded_png), "webp", quality

    try:
        # Compress using cwebp
        subprocess.run(
            ["cwebp", "-m", "6", "-q", str(quality), input_png, "-o", str(output_path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Decode back to PNG for consistent comparison
        subprocess.run(
            ["dwebp", str(output_path), "-o", str(decoded_png)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        print(f"Compressed image: {output_path} (webp, quality={quality})")
        return str(output_path), str(decoded_png), "webp", quality

    except Exception as e:
        print(f"Error compressing image with WebP: {e}")
        raise


def encode_avif(input_png, compressed_dir, decoded_dir, quality=None):
    """Encode image to AVIF format using avifenc.

    Args:
        input_png (str): Path to input PNG image
        compressed_dir (Path): Directory for compressed output
        decoded_dir (Path): Directory for decoded output
        quality (int, optional): Quality setting (default: random 5-95)

    Returns:
        tuple: (compressed_path, decoded_path, encoder_type, quality)
    """
    # Create output directories if they don't exist
    compressed_dir = Path(compressed_dir)
    decoded_dir = Path(decoded_dir)
    compressed_dir.mkdir(parents=True, exist_ok=True)
    decoded_dir.mkdir(parents=True, exist_ok=True)

    img_path = Path(input_png)
    image_id = img_path.stem

    # If quality is not provided, select randomly
    if quality is None:
        quality = random.randint(5, 95)

    output_path = Path(compressed_dir) / f"{image_id}_avif_{quality}.avif"
    decoded_png = Path(decoded_dir) / f"{image_id}_avif_{quality}_decoded.png"

    # Skip compression if file already exists (from a previous run)
    if output_path.exists() and decoded_png.exists():
        return str(output_path), str(decoded_png), "avif", quality

    try:
        # Compress using avifenc
        subprocess.run(
            [
                "avifenc",
                "-q",
                str(quality),
                "-j",
                "all",
                "-s",
                "0",
                input_png,
                str(output_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Decode back to PNG for consistent comparison
        subprocess.run(
            ["avifdec", str(output_path), str(decoded_png)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        print(f"Compressed image: {output_path} (avif, quality={quality})")
        return str(output_path), str(decoded_png), "avif", quality

    except Exception as e:
        print(f"Error compressing image with AVIF: {e}")
        raise
