"""Core benchmark functionality for press_it."""

import os
import time
import random
import signal
import csv
from pathlib import Path

from press_it.benchmark.image import (
    get_random_image,
    encode_mozjpeg,
    encode_webp,
    encode_avif,
)
from press_it.benchmark.engines import (
    run_python_ssimulacra2,
    run_cpp_ssimulacra2,
    run_rust_ssimulacra2,
)


class BenchmarkRunner:
    """Class to manage a benchmark run."""

    def __init__(self, output_file=None, temp_dir=None, quality_min=5, quality_max=95):
        """Initialize the benchmark runner.

        Args:
            output_file (str, optional): Path to save results CSV
            temp_dir (str, optional): Path to use for temporary files
            quality_min (int): Minimum quality value to test (5-100)
            quality_max (int): Maximum quality value to test (5-100)
        """
        # Set quality range
        self.quality_min = max(5, min(100, quality_min))
        self.quality_max = max(5, min(100, quality_max))

        if self.quality_min > self.quality_max:
            self.quality_min, self.quality_max = self.quality_max, self.quality_min
        # Set up directories
        self.temp_dir = Path(temp_dir or "./benchmark_temp")
        self.results_dir = Path(self.temp_dir.parent / "benchmark_results")
        self.original_image_dir = self.temp_dir / "originals"
        self.compressed_image_dir = self.temp_dir / "compressed"
        self.decoded_dir = self.temp_dir / "decoded"

        # Create directories if they don't exist
        for dir_path in [
            self.temp_dir,
            self.results_dir,
            self.original_image_dir,
            self.compressed_image_dir,
            self.decoded_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Set up output file
        if output_file:
            self.output_file = Path(output_file)
        else:
            # Default output file with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.output_file = (
                self.results_dir / f"ssimulacra2_benchmark_{timestamp}.csv"
            )

        # Results storage
        self.results = []
        self.running = True

        # Set up signal handler for graceful termination
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, sig, frame):
        """Handle Ctrl+C to gracefully stop the script."""
        print("\nStopping... Please wait for current operation to complete.")
        self.running = False

    def compress_image(self, original_path):
        """Compress the image using a random compression method and quality level.

        Args:
            original_path (str): Path to the original image

        Returns:
            tuple: (compressed_path, decoded_path, compression_type, quality)
        """
        # Choose a random encoder
        encoder_funcs = [encode_mozjpeg, encode_webp, encode_avif]
        encoder = random.choice(encoder_funcs)

        # Apply random quality from configured range
        quality = random.randint(self.quality_min, self.quality_max)

        try:
            return encoder(
                original_path, self.compressed_image_dir, self.decoded_dir, quality
            )
        except Exception as e:
            print(f"Compression failed, trying MozJPEG as fallback: {e}")
            # Fallback to MozJPEG if the chosen encoder fails
            return encode_mozjpeg(
                original_path, self.compressed_image_dir, self.decoded_dir, quality
            )

    def evaluate_image(
        self, original_path, decoded_path, compressed_path, compression_type, quality
    ):
        """Evaluate the image using all three SSIMULACRA2 implementations.

        Args:
            original_path (str): Path to the original image
            decoded_path (str): Path to the decoded image
            compressed_path (str): Path to the compressed image
            compression_type (str): Compression format
            quality (int): Compression quality setting

        Returns:
            dict: Result data for this image
        """
        print(f"Evaluating: {Path(compressed_path).name}")

        # Get file size information
        original_size = os.path.getsize(original_path)
        compressed_size = os.path.getsize(compressed_path)
        compression_ratio = (
            original_size / compressed_size if compressed_size > 0 else float("inf")
        )

        # Get image dimensions
        from PIL import Image

        img = Image.open(original_path)
        width, height = img.size

        # Run evaluations
        python_score = None
        cpp_score = None
        rust_score = None

        # Python implementation
        try:
            python_score = run_python_ssimulacra2(original_path, decoded_path)
            print(f"  Python score: {python_score}")
        except Exception as e:
            print(f"Error with Python implementation: {e}")

        # C++ implementation
        try:
            cpp_score = run_cpp_ssimulacra2(original_path, decoded_path)
            if cpp_score is not None:
                print(f"  C++ score: {cpp_score}")
            else:
                print("  C++ score: N/A")
        except Exception as e:
            print(f"Error with C++ implementation: {e}")

        # Rust implementation
        try:
            rust_score = run_rust_ssimulacra2(original_path, decoded_path)
            if rust_score is not None:
                print(f"  Rust score: {rust_score}")
            else:
                print("  Rust score: N/A")
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
            "compression_type": compression_type,
            "quality": quality,
            "python_score": python_score,
            "cpp_score": cpp_score,
            "rust_score": rust_score,
        }

    def save_results(self):
        """Save the collected results to CSV file."""
        if not self.results:
            print("No results to save.")
            return

        # Define CSV headers from the first result's keys
        headers = self.results[0].keys()

        # Write to CSV
        with open(self.output_file, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            writer.writerows(self.results)

        print(f"Results saved to {self.output_file}")

    def run(self, num_images=0):
        """Run the benchmark.

        Args:
            num_images (int): Number of images to process (0 for infinite until Ctrl+C)

        Returns:
            int: Number of images processed
        """
        # Determine run mode (infinite or fixed number)
        infinite_mode = num_images <= 0

        print(
            f"Starting SSIMULACRA2 benchmark. {'Press Ctrl+C to stop' if infinite_mode else f'Processing {num_images} images'}."
        )
        print(f"Images will be saved to: {self.original_image_dir}")
        print(f"Compressed versions will be saved to: {self.compressed_image_dir}")
        print(f"Decoded images will be saved to: {self.decoded_dir}")
        print(f"Results will be saved to: {self.output_file}")

        try:
            image_count = 0
            while self.running and (infinite_mode or image_count < num_images):
                try:
                    # Get a random original image
                    original_path = get_random_image(
                        self.original_image_dir, image_count
                    )

                    # Compress with random format and quality
                    compressed_path, decoded_path, compression_type, quality = (
                        self.compress_image(original_path)
                    )

                    # Evaluate with all implementations
                    result = self.evaluate_image(
                        original_path,
                        decoded_path,
                        compressed_path,
                        compression_type,
                        quality,
                    )
                    self.results.append(result)

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
            if self.results:
                self.save_results()
                print(f"\nBenchmark complete. Processed {len(self.results)} images.")
            else:
                print("\nNo results collected.")

            return image_count
