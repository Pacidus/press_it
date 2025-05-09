"""Benchmark runner for press_it."""

import os
import time
import signal
import shutil
import tempfile
import random  # Added missing import
import pandas as pd
from pathlib import Path
import atexit

from press_it.utils.image import get_image_dimensions
from press_it.benchmark.image import get_random_image, encode_random_format
from press_it.benchmark.engines import (
    run_python_ssimulacra2,
    run_cpp_ssimulacra2,
    run_rust_ssimulacra2,
    PYTHON_SSIMULACRA2_VERSION,
    CPP_SSIMULACRA2_VERSION,
    RUST_SSIMULACRA2_VERSION,
    PYTHON_SSIMULACRA2_AVAILABLE,
    CPP_SSIMULACRA2_AVAILABLE,
    RUST_SSIMULACRA2_AVAILABLE,
)
from press_it import __version__ as PRESS_IT_VERSION


def setup_signal_handler(callback):
    """Set up signal handler for graceful termination.

    Args:
        callback: Function to call when SIGINT is received

    Returns:
        function: The original handler
    """
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(sig, frame):
        callback()

        # Call original handler if not default
        if original_handler not in (signal.SIG_DFL, signal.SIG_IGN):
            original_handler(sig, frame)

    signal.signal(signal.SIGINT, handler)
    return original_handler


def register_cleanup(cleanup_func):
    """Register a function to be called at exit.

    Args:
        cleanup_func: Function to call at exit
    """
    atexit.register(cleanup_func)


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
        target_count=0,
    ):
        """Initialize the benchmark runner.

        Args:
            output_file: Path to save results Parquet
            temp_dir: Path to use for temporary files
            quality_min: Minimum quality value to test (5-100)
            quality_max: Maximum quality value to test (5-100)
            verbose: Whether to show detailed progress
            keep_images: Whether to keep temp images after run
            target_count: Number of images to process (0 for infinite)
        """
        # Set quality range
        self.quality_min = max(5, min(100, quality_min))
        self.quality_max = max(5, min(100, quality_max))
        self.verbose = verbose
        self.keep_images = keep_images
        self.target_count = target_count

        # Fix quality range if needed
        if self.quality_min > self.quality_max:
            self.quality_min, self.quality_max = self.quality_max, self.quality_min

        # Set up temporary directory
        if temp_dir:
            self.temp_dir = Path(temp_dir)
            self.auto_clean_temp = False
        else:
            self.system_temp_dir = tempfile.TemporaryDirectory(
                prefix="press_it_benchmark_"
            )
            self.temp_dir = Path(self.system_temp_dir.name)
            self.auto_clean_temp = True

        # Create subdirectories
        self._setup_directories()

        # Set up output file
        self._setup_output_file(output_file)

        # Results storage and state
        self.results = []
        self.running = True

        # Set up signal handler
        self._setup_signal_handler()

        # Register cleanup if needed
        if not self.keep_images and not self.auto_clean_temp:
            register_cleanup(self._cleanup_temp_files)

    def _setup_directories(self):
        """Set up directories for benchmark files."""
        self.original_dir = self.temp_dir / "originals"
        self.compressed_dir = self.temp_dir / "compressed"
        self.decoded_dir = self.temp_dir / "decoded"

        # Create directories
        self.original_dir.mkdir(parents=True, exist_ok=True)
        self.compressed_dir.mkdir(parents=True, exist_ok=True)
        self.decoded_dir.mkdir(parents=True, exist_ok=True)

    def _setup_output_file(self, output_file):
        """Set up output file path."""
        if output_file:
            self.output_file = Path(output_file)
        else:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.output_file = Path(f"ssimulacra2_benchmark_{timestamp}.parquet")

    def _setup_signal_handler(self):
        """Set up signal handler for graceful termination."""

        def stop_running():
            print("\nStopping... Please wait for current operation to complete.")
            self.running = False

        setup_signal_handler(stop_running)

    def _cleanup_temp_files(self):
        """Clean up temporary files."""
        if self.verbose:
            print("Cleaning up temporary files...")

        try:
            if self.original_dir.exists():
                shutil.rmtree(self.original_dir)
            if self.compressed_dir.exists():
                shutil.rmtree(self.compressed_dir)
            if self.decoded_dir.exists():
                shutil.rmtree(self.decoded_dir)

            # Try to remove temp dir if empty
            if self.temp_dir.exists() and not any(self.temp_dir.iterdir()):
                os.rmdir(self.temp_dir)

        except Exception as e:
            print(f"Error during cleanup: {e}")

    def process_image(self, image_id):
        """Process a single image with random compression settings.

        Args:
            image_id: Image identifier

        Returns:
            dict: Benchmark result data
        """
        # Get a random image
        original_path = get_random_image(self.original_dir, image_id, self.verbose)

        # Compress with random format and quality
        compressed_path, decoded_path, compression_type, quality = encode_random_format(
            original_path,
            self.compressed_dir,
            self.decoded_dir,
            quality=random.randint(self.quality_min, self.quality_max),
        )

        # Get file sizes
        original_size = os.path.getsize(original_path)
        compressed_size = os.path.getsize(compressed_path)
        compression_ratio = (
            original_size / compressed_size if compressed_size > 0 else float("inf")
        )

        # Get image dimensions
        width, height = get_image_dimensions(original_path)

        # Run quality evaluations
        python_score = run_python_ssimulacra2(original_path, decoded_path)
        cpp_score = run_cpp_ssimulacra2(original_path, decoded_path)
        rust_score = run_rust_ssimulacra2(original_path, decoded_path)

        # Print summary
        print(
            f"Format: {compression_type}, Quality: {quality}, Size: {compressed_size} bytes"
        )
        print(f"Compression ratio: {compression_ratio:.2f}x")
        print(f"Python score: {python_score or 'N/A'}")
        print(f"C++ score: {cpp_score or 'N/A'}")
        print(f"Rust score: {rust_score or 'N/A'}")

        # Return result
        return {
            "timestamp": pd.Timestamp.now(),
            "press_it_version": PRESS_IT_VERSION,
            "python_ssimulacra2_version": PYTHON_SSIMULACRA2_VERSION,
            "cpp_ssimulacra2_version": CPP_SSIMULACRA2_VERSION,
            "rust_ssimulacra2_version": RUST_SSIMULACRA2_VERSION,
            "original_path": str(original_path),
            "compressed_path": str(compressed_path),
            "decoded_path": str(decoded_path),
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
        """Save collected results to Parquet file."""
        if not self.results:
            print("No results to save.")
            return

        # Ensure parent directory exists
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        # Convert to DataFrame
        new_results_df = pd.DataFrame(self.results)

        # Append to existing file if it exists
        if self.output_file.exists():
            try:
                existing_df = pd.read_parquet(self.output_file)
                combined_df = pd.concat(
                    [existing_df, new_results_df], ignore_index=True
                )

                if self.verbose:
                    print(
                        f"Appending {len(new_results_df)} rows to existing {len(existing_df)} rows"
                    )

                combined_df.to_parquet(
                    self.output_file,
                    engine="pyarrow",
                    compression="snappy",
                    index=False,
                )

            except Exception as e:
                print(f"Error appending to existing file: {e}")
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

        # Clear results list
        self.results = []

    def run(self):
        """Run the benchmark.

        Returns:
            int: Number of images processed
        """
        # Print information
        self._print_benchmark_info()

        try:
            image_count = 0
            batch_count = 0

            # Main benchmark loop
            while self.running and (
                self.target_count <= 0 or image_count < self.target_count
            ):
                try:
                    print(
                        f"\nProcessing image {image_count + 1}",
                        "" if self.target_count <= 0 else f"/{self.target_count}",
                    )

                    # Process image
                    result = self.process_image(image_count)
                    self.results.append(result)

                    # Update counters
                    image_count += 1
                    batch_count += 1

                    # Save periodically
                    if batch_count >= 5:
                        if self.verbose:
                            print(f"Saving batch of {batch_count} results...")
                        self.save_results()
                        batch_count = 0

                    # Print separator
                    print("-" * 40)
                    print(f"Processed {image_count} images")

                    # Small delay to prevent system overload
                    time.sleep(0.5)

                except KeyboardInterrupt:
                    break

                except Exception as e:
                    print(f"Error processing image: {e}")
                    time.sleep(1)  # Wait before retrying

        finally:
            # Save any remaining results
            if self.results:
                print(f"\nSaving final batch of {len(self.results)} results...")
                self.save_results()

            print(f"\nBenchmark complete. Processed {image_count} images.")

            # Clean up if needed
            if self.auto_clean_temp:
                try:
                    self.system_temp_dir.cleanup()
                except Exception as e:
                    print(f"Error cleaning up temporary directory: {e}")

            return image_count

    def _print_benchmark_info(self):
        """Print benchmark information."""
        print(f"Starting SSIMULACRA2 benchmark")
        print(f"Results will be saved to: {self.output_file}")
        print(f"Using temporary directory: {self.temp_dir}")

        # Print version information
        print("\nImplementation Versions:")
        print(f"Press-it: {PRESS_IT_VERSION}")
        print(
            f"Python SSIMULACRA2: {PYTHON_SSIMULACRA2_VERSION or 'Not available'} "
            f"({'Available' if PYTHON_SSIMULACRA2_AVAILABLE else 'Not available'})"
        )
        print(
            f"C++ SSIMULACRA2: {CPP_SSIMULACRA2_VERSION or 'Not available'} "
            f"({'Available' if CPP_SSIMULACRA2_AVAILABLE else 'Not available'})"
        )
        print(
            f"Rust SSIMULACRA2: {RUST_SSIMULACRA2_VERSION or 'Not available'} "
            f"({'Available' if RUST_SSIMULACRA2_AVAILABLE else 'Not available'})"
        )

        # Print settings
        print(f"\nQuality range: {self.quality_min} to {self.quality_max}")
        print(f"Verbose mode: {'Enabled' if self.verbose else 'Disabled'}")
        print(f"Keep images: {'Yes' if self.keep_images else 'No'}")

        if self.target_count <= 0:
            print("Running in infinite mode (press Ctrl+C to stop)")
        else:
            print(f"Target image count: {self.target_count}")
