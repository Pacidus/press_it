"""Image optimizer targeting specific SSIM values using multiple encoders."""

import argparse
import os
import shutil
import subprocess
import tempfile

from tqdm import tqdm

FILE_EXTENSIONS = {
    "mozjpeg": "jpg",
    "webp": "webp",
    "avif": "avif",
    "png": "png",
}

SSIM_METHODS = {
    0: "ImageMagick",
    1: "as2c",
}

DEPENDENCIES = [
    "pngcrush",
    "avifenc",
    "avifdec",
    "magick",
    "cjpeg",
    "cwebp",
    "dwebp",
]


class MissingDependencyError(RuntimeError):
    """Custom exception for missing required dependencies."""


def check_dependencies(required_tools):
    """Verify required system utilities are available."""
    missing = []
    for tool in required_tools:
        if not shutil.which(tool):
            missing.append(tool)

    if missing:
        raise MissingDependencyError(
            f"Missing required tools: {', '.join(missing)}. "
            "Please install them before running this script."
        )


def resize(file, size):
    """Resize the file to the size."""
    subprocess.run(
        [
            "magick",
            file,
            "-resize",
            size,
            file,
        ],
        check=True,
    )


def calculate_ssim_imagemagick(original_path, compressed_path):
    """Calculate SSIM using ImageMagick's compare tool.

    Returns SSIM value between 0-100
    """
    result = subprocess.run(
        ["compare", "-metric ssim", original_path, compressed_path, "null:"],
        capture_output=True,
        text=True,
    )
    # Extract SSIM value from stderr output
    ssim_str = result.stderr.split("(")[1].split(")")[0]
    return float(ssim_str) * 100


def calculate_ssim_as2c(original_path, compressed_path):
    """Calculate SSIM using as2c utility.

    Returns SSIM value between 0-1
    """
    try:
        result = subprocess.run(
            ["as2c", original_path, compressed_path],
            capture_output=True,
            check=True,
            text=True,
        )
        return float(result.stdout)
    except (subprocess.CalledProcessError, ValueError) as e:
        print(f"SSIM calculation failed: {str(e)}")
        return None


get_ssim = calculate_ssim_imagemagick


def quality_optimizer(encoder_func):
    """Convert encoder function into binary search.

    Find optimal quality setting that meets target SSIM requirement
    """

    def binary_search(original_path, temp_dir, target_ssim):
        best = (None, None)
        extension = FILE_EXTENSIONS[encoder_func.__name__.split("_")[1]]
        base_name = os.path.splitext(os.path.basename(args.input_image))[0]
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
                current_ssim = get_ssim(original_path, decoded_png)
                if current_ssim >= target_ssim:
                    best = (mid, os.path.getsize(output_path))
                    mid -= pos
                pbar.set_description(f"{extension: >5} {mid: 5d} {current_ssim: 5.1f}")
            except subprocess.CalledProcessError as e:
                print(f"Encoder failed at quality {mid}: {str(e)}")
                continue

        return best

    return binary_search


@quality_optimizer
def encode_webp(input_png, output_path, quality):
    """Encode image to WebP format using cwebp."""
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


@quality_optimizer
def encode_avif(input_png, output_path, quality):
    """Encode image to AVIF format using avifenc."""
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


@quality_optimizer
def encode_mozjpeg(input_png, output_path, quality):
    """Encode image to JPEG using MozJPEG."""
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


def optimize_png(original_path, temp_dir, target_ssim=None):
    """Optimize PNG using pngcrush."""
    base_name = os.path.splitext(os.path.basename(args.input_image))[0]
    output_path = os.path.join(temp_dir, f"{base_name}_100.png")
    subprocess.run(
        ["pngcrush", original_path, output_path],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return 100, os.path.getsize(output_path)  # Quality is meaningless for PNG


encodes = {
    "mozjpeg": encode_mozjpeg,
    "webp": encode_webp,
    "avif": encode_avif,
    "png": optimize_png,
}

if __name__ == "__main__":
    # Configure SSIM calculation method
    parser = argparse.ArgumentParser(
        description="Optimize images for target SSIM using multiple encoders",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_image",
        help="Path to source image file"
    )
    parser.add_argument(
        "target_ssim",
        type=float,
        help="Target SSIM value (0-100)"
    )
    parser.add_argument(
        "--ssim-method",
        type=int,
        choices=[0, 1],
        default=0,
        help="SSIM calculation method: 0=ImageMagick (ssim), 1=as2c (assim2)",
    )
    parser.add_argument(
        "--resize",
        "-r",
        type=str,
        help=""
        "Resize the image (width)x(height)"
        "if one of the dimentions is not set the aspect ratio is respected",
    )
    args = parser.parse_args()

    # Set global SSIM function based on user choice
    if args.ssim_method:
        get_ssim = calculate_ssim_as2c

    # Verify system dependencies
    try:
        check_dependencies(DEPENDENCIES)
    except MissingDependencyError as e:
        print(str(e))
        exit

    results = {}
    with tempfile.TemporaryDirectory() as temp_dir:
        # Convert input to PNG for consistent comparison
        reference_png = os.path.join(temp_dir, "reference.png")
        subprocess.run(["magick", args.input_image, "-alpha", "off", reference_png], check=True)
        if args.resize:
            resize(reference_png, args.resize)

        for key in encodes:
            # Process through all encoders
            encd = encodes[key]
            try:
                quality, size = encd(reference_png, temp_dir, args.target_ssim)
                print(f"best {key}: size={size}, quality={quality}")
                if quality:
                    results[key] = {
                        "size": size,
                        "quality": quality,
                    }
            except Exception as e:
                print(f"{key} encoding failed: {str(e)}")

        if not results:
            print("No successful encodings achieved")
            exit

        # Determine best format by size
        best_format = min(results, key=lambda x: results[x]["size"])
        best_result = results[best_format]

        # Create output filename
        base_name = os.path.splitext(os.path.basename(args.input_image))[0]
        output_file = (
            f"{base_name}_{best_result['quality']}"
            f".{FILE_EXTENSIONS[best_format]}"
        )

        # Move best result to current directory
        source_path = os.path.join(temp_dir, output_file)
        shutil.move(source_path, output_file)

        print(
            "\n"
            "Optimization complete!\n"
            f"    Best format: {best_format} ({best_result['size']} bytes)\n"
            f"    Output file: {output_file}"
        )
