#!/usr/bin/env python3
"""
SSIMULACRA2 Benchmark Tool with Multiple Compression Formats

This script:
1. Downloads random images from Picsum Photos
2. Compresses them using multiple formats (MozJPEG, WebP, AVIF)
3. Evaluates the compressed images with SSIMULACRA2 implementations (Python, C++, Rust)
4. Runs until the specified number of images are processed or until interrupted with Ctrl+C
5. Saves the collected data
"""

import os
import sys
import time
import random
import subprocess
import signal
import argparse
import csv
import requests
from PIL import Image
from pathlib import Path
import tempfile
import shutil

# Import Python implementation of SSIMULACRA2
try:
    from ssimulacra2 import compute_ssimulacra2_with_alpha

    SSIMULACRA2_AVAILABLE = True
except ImportError:
    print(
        "Warning: Could not import ssimulacra2 module. Python implementation will be skipped."
    )
    SSIMULACRA2_AVAILABLE = False

# Configuration
TEMP_DIR = Path("./benchmark_temp")
RESULTS_DIR = Path("./benchmark_results")
ORIGINAL_IMAGE_DIR = TEMP_DIR / "originals"
COMPRESSED_IMAGE_DIR = TEMP_DIR / "compressed"
DECODED_DIR = TEMP_DIR / "decoded"
OUTPUT_FILE = (
    RESULTS_DIR / f"ssimulacra2_benchmark_{time.strftime('%Y%m%d_%H%M%S')}.csv"
)

# Global variables
results = []
running = True

# Check for required dependencies
DEPENDENCIES = [
    "magick",
    "cjpeg",
    "cwebp",
    "dwebp",
    "avifenc",
    "avifdec",
]

# File extensions mapping
FILE_EXTENSIONS = {
    "mozjpeg": "jpg",
    "webp": "webp",
    "avif": "avif",
}


def check_dependencies(required_tools):
    """Verify required system utilities are available."""
    missing = []
    for tool in required_tools:
        if not shutil.which(tool):
            missing.append(tool)

    if missing:
        print(
            f"Missing required tools: {', '.join(missing)}. "
            "Please install them before running this script."
        )
        sys.exit(1)


def setup_directories():
    """Create necessary directories if they don't exist."""
    for dir_path in [
        TEMP_DIR,
        RESULTS_DIR,
        ORIGINAL_IMAGE_DIR,
        COMPRESSED_IMAGE_DIR,
        DECODED_DIR,
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)


def signal_handler(sig, frame):
    """Handle Ctrl+C to gracefully stop the script."""
    global running
    print("\nStopping... Please wait for current operation to complete.")
    running = False


def get_random_image():
    """Download a random image from Picsum Photos with random dimensions."""
    image_id = f"image_{len(results):04d}"
    output_path = ORIGINAL_IMAGE_DIR / f"{image_id}.png"

    # Skip download if file already exists (from a previous run)
    if output_path.exists():
        return str(output_path)

    try:
        # Get a random image with random dimensions
        width = random.choice([800, 1024, 1280, 1600])
        height = random.choice([600, 768, 960, 1200])

        # Simple URL for random image with specified dimensions
        image_url = f"https://picsum.photos/{width}/{height}"

        response = requests.get(image_url, timeout=10)
        response.raise_for_status()

        # Save the image as PNG using ImageMagick for consistency
        temp_jpg = TEMP_DIR / f"temp_{image_id}.jpg"
        with open(temp_jpg, "wb") as f:
            f.write(response.content)

        # Convert to PNG with alpha removed for consistent comparison
        subprocess.run(
            ["magick", str(temp_jpg), "-alpha", "off", str(output_path)],
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
        fallback_images = list(ORIGINAL_IMAGE_DIR.glob("*.png"))
        if fallback_images:
            fallback_path = random.choice(fallback_images)
            print(f"Using fallback image: {fallback_path}")
            return str(fallback_path)

        # If no fallback available, raise the exception
        raise


def encode_mozjpeg(input_png, quality=None):
    """Encode image to JPEG using MozJPEG."""
    img_path = Path(input_png)
    image_id = img_path.stem

    # If quality is not provided, select randomly
    if quality is None:
        quality = random.randint(5, 95)

    output_path = COMPRESSED_IMAGE_DIR / f"{image_id}_mozjpeg_{quality}.jpg"
    decoded_png = DECODED_DIR / f"{image_id}_mozjpeg_{quality}_decoded.png"

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


def encode_webp(input_png, quality=None):
    """Encode image to WebP format using cwebp."""
    img_path = Path(input_png)
    image_id = img_path.stem

    # If quality is not provided, select randomly
    if quality is None:
        quality = random.randint(5, 95)

    output_path = COMPRESSED_IMAGE_DIR / f"{image_id}_webp_{quality}.webp"
    decoded_png = DECODED_DIR / f"{image_id}_webp_{quality}_decoded.png"

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


def encode_avif(input_png, quality=None):
    """Encode image to AVIF format using avifenc."""
    img_path = Path(input_png)
    image_id = img_path.stem

    # If quality is not provided, select randomly
    if quality is None:
        quality = random.randint(5, 95)

    output_path = COMPRESSED_IMAGE_DIR / f"{image_id}_avif_{quality}.avif"
    decoded_png = DECODED_DIR / f"{image_id}_avif_{quality}_decoded.png"

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


def compress_image(original_path):
    """Compress the image using a random compression method and quality level."""
    # Choose a random encoder
    encoder_funcs = [encode_mozjpeg, encode_webp, encode_avif]
    encoder = random.choice(encoder_funcs)

    # Apply random quality
    quality = random.randint(5, 95)

    try:
        return encoder(original_path, quality)
    except Exception as e:
        print(f"Compression failed, trying MozJPEG as fallback: {e}")
        # Fallback to MozJPEG if the chosen encoder fails
        return encode_mozjpeg(original_path, quality)


def run_python_ssimulacra2(original_path, compressed_path):
    """Run the Python implementation of SSIMULACRA2."""
    try:
        if SSIMULACRA2_AVAILABLE:
            return compute_ssimulacra2_with_alpha(original_path, compressed_path)
        else:
            return None
    except Exception as e:
        print(f"Error running Python implementation: {e}")
        return None


def run_cpp_ssimulacra2(original_path, compressed_path):
    """Run the C++ implementation of SSIMULACRA2."""
    try:
        cmd = ["cmulacra2", original_path, compressed_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        print(f"Error running C++ implementation: {e}")
        print(f"stderr: {e.stderr}")
        return None
    except Exception as e:
        print(f"Unexpected error with C++ implementation: {e}")
        return None


def run_rust_ssimulacra2(original_path, compressed_path):
    """Run the Rust implementation (as2c) of SSIMULACRA2."""
    try:
        cmd = ["as2c", original_path, compressed_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        print(f"Error running Rust implementation: {e}")
        print(f"stderr: {e.stderr}")
        return None
    except Exception as e:
        print(f"Unexpected error with Rust implementation: {e}")
        return None


def evaluate_image(original_path, decoded_path, compressed_path):
    """Evaluate the image using all three SSIMULACRA2 implementations."""
    print(f"Evaluating: {Path(compressed_path).name}")

    # Get file size information
    original_size = os.path.getsize(original_path)
    compressed_size = os.path.getsize(compressed_path)
    compression_ratio = (
        original_size / compressed_size if compressed_size > 0 else float("inf")
    )

    # Get image dimensions
    img = Image.open(original_path)
    width, height = img.size

    # Run evaluations
    python_score = None
    cpp_score = None
    rust_score = None

    # Python implementation
    if SSIMULACRA2_AVAILABLE:
        try:
            python_score = run_python_ssimulacra2(original_path, decoded_path)
            print(f"  Python score: {python_score:.4f}")
        except Exception as e:
            print(f"Error with Python implementation: {e}")

    # C++ implementation
    try:
        cpp_score = run_cpp_ssimulacra2(original_path, decoded_path)
        print(f"  C++ score: {cpp_score:.4f}")
    except Exception as e:
        print(f"Error with C++ implementation: {e}")

    # Rust implementation
    try:
        rust_score = run_rust_ssimulacra2(original_path, decoded_path)
        print(f"  Rust score: {rust_score:.4f}")
    except Exception as e:
        print(f"Error with Rust implementation: {e}")

    return {
        "original_path": original_path,
        "decoded_path": decoded_path,
        "compressed_path": compressed_path,
        "width": width,
        "height": height,
        "original_size": original_size,
        "compressed_size": compressed_size,
        "compression_ratio": compression_ratio,
        "python_score": python_score,
        "cpp_score": cpp_score,
        "rust_score": rust_score,
    }


def save_results(results_data):
    """Save the collected results to CSV file."""
    if not results_data:
        print("No results to save.")
        return

    # Define CSV headers from the first result's keys
    headers = results_data[0].keys()

    # Write to CSV
    with open(OUTPUT_FILE, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results_data)

    print(f"Results saved to {OUTPUT_FILE}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark different compression formats with SSIMULACRA2"
    )
    parser.add_argument(
        "--num-images",
        "-n",
        type=int,
        default=0,
        help="Number of images to process (0 for infinite until Ctrl+C)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output CSV file path (default: auto-generated in benchmark_results directory)",
    )
    return parser.parse_args()


def main():
    """Main function to run the benchmark."""
    global results, running, OUTPUT_FILE

    # Parse command line arguments
    args = parse_args()

    # Set output file if specified
    if args.output:
        OUTPUT_FILE = Path(args.output)

    # Setup
    check_dependencies(DEPENDENCIES)
    setup_directories()
    signal.signal(signal.SIGINT, signal_handler)

    # Determine run mode (infinite or fixed number)
    num_images = args.num_images
    infinite_mode = num_images <= 0

    print(
        f"Starting SSIMULACRA2 benchmark. {'Press Ctrl+C to stop' if infinite_mode else f'Processing {num_images} images'}."
    )
    print(f"Images will be saved to: {ORIGINAL_IMAGE_DIR}")
    print(f"Compressed versions will be saved to: {COMPRESSED_IMAGE_DIR}")
    print(f"Decoded images will be saved to: {DECODED_DIR}")
    print(f"Results will be saved to: {OUTPUT_FILE}")

    try:
        image_count = 0
        while running and (infinite_mode or image_count < num_images):
            try:
                # Get a random original image
                original_path = get_random_image()

                # Compress with random format and quality
                compressed_path, decoded_path, compression_type, quality = (
                    compress_image(original_path)
                )

                # Evaluate with all implementations
                result = evaluate_image(original_path, decoded_path, compressed_path)
                result["compression_type"] = compression_type
                result["quality"] = quality
                results.append(result)

                # Increment counter
                image_count += 1

                # Print a separator
                print("-" * 40)
                print(
                    f"Processed {image_count}/{num_images if not infinite_mode else 'infinite'} images"
                )

                # Optional small delay to prevent system overload
                time.sleep(0.5)

            except KeyboardInterrupt:
                break

            except Exception as e:
                print(f"Error in benchmark cycle: {e}")
                time.sleep(1)  # Wait a bit before retrying

    finally:
        # Even if there's an unhandled exception, we want to save the results
        if results:
            save_results(results)
            print(f"\nBenchmark complete. Processed {len(results)} images.")
            print(f"Results saved to {OUTPUT_FILE}")
        else:
            print("\nNo results collected.")


if __name__ == "__main__":
    main()
