"""Image encoders for press_it."""

import os
import subprocess
from functools import wraps
from tqdm import tqdm


# File extensions for each encoder
FILE_EXTENSIONS = {
    "mozjpeg": "jpg",
    "webp": "webp",
    "avif": "avif",
    "png": "png",
}


def quality_optimizer(get_ssim_func):
    """Decorator to convert encoder function into binary search.

    Args:
        get_ssim_func (callable): Function to calculate SSIM

    Returns:
        callable: Decorated function that performs binary search for optimal quality
    """

    def decorator(encoder_func):
        @wraps(encoder_func)
        def binary_search(original_path, temp_dir, target_ssim, input_image):
            best = (None, None)
            extension = FILE_EXTENSIONS[encoder_func.__name__.split("_")[1]]
            base_name = os.path.splitext(os.path.basename(input_image))[0]
            # Binary search with 7 iterations
            itt = list(range(7))[::-1]
            mid = 0
            for i in (pbar := tqdm(itt, desc=f"{extension: >5}  None  None")):
                if mid == 100:
                    continue
                else:
                    pos = 0b1 << i
                    if (mid := mid + pos) > 100:
                        mid = 100
                output_path = os.path.join(temp_dir, f"{base_name}_{mid}.{extension}")
                try:
                    decoded_png = encoder_func(original_path, output_path, mid)
                    current_ssim = get_ssim_func(original_path, decoded_png)
                    if current_ssim >= target_ssim:
                        best = (mid, os.path.getsize(output_path))
                        mid -= pos
                    pbar.set_description(
                        f"{extension: >5} {mid: 5d} {current_ssim: 5.1f}"
                    )
                except subprocess.CalledProcessError as e:
                    print(f"Encoder failed at quality {mid}: {str(e)}")
                    continue

            return best

        return binary_search

    return decorator


def encode_webp(input_png, output_path, quality):
    """Encode image to WebP format using cwebp.

    Args:
        input_png (str): Path to input PNG image
        output_path (str): Path for output WebP image
        quality (int): Compression quality (1-100)

    Returns:
        str: Path to decoded PNG (for quality comparison)
    """
    subprocess.run(
        ["cwebp", "-m", "6", "-q", str(quality), input_png, "-o", output_path],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    decoded_png = os.path.join(os.path.dirname(output_path), "decoded_webp.png")
    subprocess.run(
        ["dwebp", output_path, "-o", decoded_png],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return decoded_png


def encode_avif(input_png, output_path, quality):
    """Encode image to AVIF format using avifenc.

    Args:
        input_png (str): Path to input PNG image
        output_path (str): Path for output AVIF image
        quality (int): Compression quality (1-100)

    Returns:
        str: Path to decoded PNG (for quality comparison)
    """
    subprocess.run(
        ["avifenc", "-q", str(quality), "-j", "all", "-s", "0", input_png, output_path],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    decoded_png = os.path.join(os.path.dirname(output_path), "decoded_avif.png")
    subprocess.run(
        ["avifdec", output_path, decoded_png],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return decoded_png


def encode_mozjpeg(input_png, output_path, quality):
    """Encode image to JPEG using MozJPEG.

    Args:
        input_png (str): Path to input PNG image
        output_path (str): Path for output JPEG image
        quality (int): Compression quality (1-100)

    Returns:
        str: Path to decoded PNG (for quality comparison)
    """
    subprocess.run(
        [
            "cjpeg",
            "-quality",
            str(quality),
            "-outfile",
            output_path,
            input_png,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    decoded_png = os.path.join(os.path.dirname(output_path), "decoded_mozjpeg.png")
    subprocess.run(
        ["magick", output_path, decoded_png],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return decoded_png


def optimize_png(input_png, output_path, quality=None):
    """Optimize PNG using pngcrush.

    Args:
        input_png (str): Path to input PNG image
        output_path (str): Path for output optimized PNG
        quality (int, optional): Ignored for PNG

    Returns:
        str: Path to output PNG (same as input since no decoding needed)
    """
    subprocess.run(
        ["pngcrush", input_png, output_path],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return output_path
