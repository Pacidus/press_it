"""Core benchmark functionality for press_it."""

import os
import time
import random
import signal
import shutil
import pandas as pd
from pathlib import Path
import atexit

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
    PYTHON_SSIMULACRA2_VERSION,
    CPP_SSIMULACRA2_VERSION,
    RUST_SSIMULACRA2_VERSION,
    SSIMULACRA2_AVAILABLE,
    CPP_SSIMULACRA2_AVAILABLE,
    RUST_SSIMULACRA2_AVAILABLE,
)
from press_it import __version__ as PRESS_IT_VERSION


class BenchmarkRunner:
    """Class to manage a benchmark run."""

    def __init__(
        self,
        output_file=None,
        temp_dir=None,
        quality_min=5,
        quality_max=95,
        verbose=False,
        keep_images=False,
    ):
        """Initialize the benchmark runner.

        Args:
            output_file (str, optional): Path to save results Parquet
            temp_dir (str, optional): Path to use for temporary files
            quality_min (int): Minimum quality value to test (5-100)
            quality_max (int): Maximum quality value to test (5-100)
            verbose (bool): Whether to show detailed progress information
            keep_images (bool): Whether to keep image files after benchmark
        """
        # Set quality range
        self.quality_min = max(5, min(100, quality_min))
        self.quality_max = max(5, min(100, quality_max))
        self.verbose = verbose
        self.keep_images = keep_images

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
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Only create image directories when needed
        # This avoids creating empty directories

        # Set up output file
        if output_file:
            self.output_file = Path(output_file)
        else:
            # Default output file - no timestamp, just a fixed name for easier automation
            self.output_file = self.results_dir / "ssimulacra2_benchmark.parquet"

        # Results storage
        self.results = []
        self.running = True

        # Set up signal handler for graceful termination
        signal.signal(signal.SIGINT, self._signal_handler)

        # Set up cleanup handler for temporary files
        if not self.keep_images:
            atexit.register(self._cleanup_temp_files)

    def _signal_handler(self, sig, frame):
        """Handle Ctrl+C to gracefully stop the script."""
        print("\nStopping... Please wait for current operation to complete.")
        self.running = False

    def _cleanup_temp_files(self):
        """Clean up temporary image files."""
        if self.verbose:
            print("Cleaning up temporary image files...")

        # Only remove temp image directories, keep the parquet file
        try:
            if self.original_image_dir.exists():
                shutil.rmtree(self.original_image_dir)
            if self.compressed_image_dir.exists():
                shutil.rmtree(self.compressed_image_dir)
            if self.decoded_dir.exists():
                shutil.rmtree(self.decoded_dir)
        except Exception as e:
            print(f"Error during cleanup: {e}")

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
        encoder_name = encoder.__name__.split("_")[1]

        # Apply random quality from configured range
        quality = random.randint(self.quality_min, self.quality_max)

        if self.verbose:
            print(
                f"Compressing {Path(original_path).name} with {encoder_name} at quality {quality}"
            )

        try:
            return encoder(
                original_path, self.compressed_image_dir, self.decoded_dir, quality
            )
        except Exception as e:
            if self.verbose:
                print(
                    f"Compression failed with {encoder_name}, trying MozJPEG as fallback: {e}"
                )
            # Fallback to MozJPEG if the chosen encoder fails
            return encode_mozjpeg(
                original_path, self.compressed_image_dir, self.decoded_dir, quality
            )

    def evaluate_image(
        self, original_path, decoded_path, compressed_path, compression_type, quality
    ):
        """Evaluate the image using all three SSIMULACRA2 implementations.

        Args:
            original_path (str): Path to original image
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
            if self.verbose:
                print("  Running Python SSIMULACRA2 implementation...")
            python_score = run_python_ssimulacra2(original_path, decoded_path)
            print(f"  Python score: {python_score}")
        except Exception as e:
            print(f"Error with Python implementation: {e}")

        # C++ implementation
        try:
            if self.verbose:
                print("  Running C++ SSIMULACRA2 implementation...")
            cpp_score = run_cpp_ssimulacra2(original_path, decoded_path)
            if cpp_score is not None:
                print(f"  C++ score: {cpp_score}")
            else:
                print("  C++ score: N/A")
        except Exception as e:
            print(f"Error with C++ implementation: {e}")

        # Rust implementation
        try:
            if self.verbose:
                print("  Running Rust SSIMULACRA2 implementation...")
            rust_score = run_rust_ssimulacra2(original_path, decoded_path)
            if rust_score is not None:
                print(f"  Rust score: {rust_score}")
            else:
                print("  Rust score: N/A")
        except Exception as e:
            print(f"Error with Rust implementation: {e}")

        if self.verbose:
            print(
                f"  Compression ratio: {compression_ratio:.2f}x ({compressed_size} bytes)"
            )

        # Now add version information
        return {
            "timestamp": pd.Timestamp.now(),
            "press_it_version": PRESS_IT_VERSION,
            "python_ssimulacra2_version": PYTHON_SSIMULACRA2_VERSION,
            "cpp_ssimulacra2_version": CPP_SSIMULACRA2_VERSION,
            "rust_ssimulacra2_version": RUST_SSIMULACRA2_VERSION,
            "original_path": str(original_path),
            "decoded_path": str(decoded_path),
            "compressed_path": str(compressed_path),
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
        """Save the collected results to Parquet file, always appending if it exists."""
        if not self.results:
            print("No results to save.")
            return

        # Convert results to DataFrame
        new_results_df = pd.DataFrame(self.results)

        # Check if the output file already exists
        if self.output_file.exists():
            try:
                # Append new results to existing file
                existing_df = pd.read_parquet(self.output_file)
                combined_df = pd.concat(
                    [existing_df, new_results_df], ignore_index=True
                )

                if self.verbose:
                    print(
                        f"Appending {len(new_results_df)} rows to existing {len(existing_df)} rows"
                    )

                # Write combined data back to parquet with good compression
                combined_df.to_parquet(
                    self.output_file,
                    engine="pyarrow",
                    compression="snappy",
                    index=False,
                )

            except Exception as e:
                print(f"Error appending to existing parquet file: {e}")
                print("Creating new file instead.")
                new_results_df.to_parquet(
                    self.output_file,
                    engine="pyarrow",
                    compression="snappy",
                    index=False,
                )
        else:
            # Create new file
            new_results_df.to_parquet(
                self.output_file, engine="pyarrow", compression="snappy", index=False
            )

        print(f"Results saved to {self.output_file}")

        # Reset results list to avoid duplicating data on next save
        self.results = []

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
        print(f"Results will be saved to and appended to: {self.output_file}")

        # Print version information
        print("\nImplementation Versions:")
        print(f"Press-it: {PRESS_IT_VERSION}")
        print(
            f"Python SSIMULACRA2: {PYTHON_SSIMULACRA2_VERSION or 'Not available'} {'(Available)' if SSIMULACRA2_AVAILABLE else '(Not available)'}"
        )
        print(
            f"C++ SSIMULACRA2: {CPP_SSIMULACRA2_VERSION or 'Not available'} {'(Available)' if CPP_SSIMULACRA2_AVAILABLE else '(Not available)'}"
        )
        print(
            f"Rust SSIMULACRA2: {RUST_SSIMULACRA2_VERSION or 'Not available'} {'(Available)' if RUST_SSIMULACRA2_AVAILABLE else '(Not available)'}"
        )

        if self.verbose:
            print(f"Quality range: {self.quality_min} to {self.quality_max}")
            print(f"Verbose mode: Enabled")
            print(f"Keep images after benchmark: {self.keep_images}")
            print(f"Images will be saved to: {self.original_image_dir}")
            print(f"Compressed versions will be saved to: {self.compressed_image_dir}")
            print(f"Decoded images will be saved to: {self.decoded_dir}")

        try:
            image_count = 0
            batch_count = 0

            while self.running and (infinite_mode or image_count < num_images):
                try:
                    # Get a random original image
                    if self.verbose:
                        print(
                            f"\nProcessing image {image_count + 1}{'/' + str(num_images) if not infinite_mode else ''}"
                        )
                    original_path = get_random_image(
                        self.original_image_dir, image_count, self.verbose
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
                    batch_count += 1

                    # Save to Parquet periodically (every 5 images)
                    if batch_count >= 5:
                        if self.verbose:
                            print(f"Saving batch of {batch_count} results...")
                        self.save_results()
                        batch_count = 0

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
            # Save any remaining results
            if self.results:
                if self.verbose:
                    print(f"\nSaving final batch of {len(self.results)} results...")
                self.save_results()
                print(f"\nBenchmark complete. Processed {image_count} images in total.")
            else:
                print("\nNo results collected.")

            return image_count
