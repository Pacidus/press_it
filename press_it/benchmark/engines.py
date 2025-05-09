"""SSIMULACRA2 engine wrappers for press_it benchmarks."""

import subprocess
import re
import importlib.metadata
import sys
import tempfile
from pathlib import Path
from PIL import Image

# Try to import the Python implementation of SSIMULACRA2
try:
    from ssimulacra2 import compute_ssimulacra2_with_alpha

    SSIMULACRA2_AVAILABLE = True

    # Try to get the version of the Python implementation
    try:
        # Try using importlib.metadata first (Python 3.8+)
        PYTHON_SSIMULACRA2_VERSION = importlib.metadata.version("ssimulacra2")
    except (importlib.metadata.PackageNotFoundError, AttributeError):
        # Fallback: try to extract from module if available
        try:
            import ssimulacra2

            if hasattr(ssimulacra2, "__version__"):
                PYTHON_SSIMULACRA2_VERSION = ssimulacra2.__version__
            else:
                PYTHON_SSIMULACRA2_VERSION = "unknown"
        except (ImportError, AttributeError):
            PYTHON_SSIMULACRA2_VERSION = "unknown"
except ImportError:
    print(
        "Warning: Could not import ssimulacra2 module. Python implementation will be skipped."
    )
    SSIMULACRA2_AVAILABLE = False
    PYTHON_SSIMULACRA2_VERSION = None


def get_cpp_ssimulacra2_version():
    """Get the version of the C++ SSIMULACRA2 implementation.

    Returns:
        str: Version string or None if not available
    """
    try:
        # Check if executable exists
        which_result = subprocess.run(
            ["which", "ssimulacra2"], capture_output=True, text=True, check=False
        )
        if which_result.returncode != 0:
            return None

        # Check basic functionality by running without arguments
        # The version is in stderr for this implementation
        version_result = subprocess.run(
            ["ssimulacra2"], capture_output=True, text=True, check=False
        )

        # Verify functionality with test images
        functional = False
        with tempfile.TemporaryDirectory() as temp_dir:
            test_img1 = Path(temp_dir) / "test1.png"
            test_img2 = Path(temp_dir) / "test2.png"

            # Create white images
            Image.new("RGB", (10, 10), color="white").save(test_img1)
            Image.new("RGB", (10, 10), color="white").save(test_img2)

            # Run test
            test_result = subprocess.run(
                ["ssimulacra2", str(test_img1), str(test_img2)],
                capture_output=True,
                text=True,
                check=False,
            )

            # Check if it produced a valid score
            if test_result.returncode == 0 and test_result.stdout.strip():
                try:
                    float(test_result.stdout.strip())
                    functional = True
                except ValueError:
                    pass

        if not functional:
            return None  # Not working properly

        # Extract version from stderr (based on diagnostic output)
        if version_result.stderr:
            # Look for pattern "SSIMULACRA 2.1 [AVX3_ZEN4,AVX3,AVX2,SSE4,SSSE3]"
            lines = version_result.stderr.strip().split("\n")
            if lines:
                first_line = lines[0]
                version_match = re.search(r"SSIMULACRA\s+(\d+\.\d+)", first_line)
                if version_match:
                    return version_match.group(1)

        # It's working but version couldn't be determined
        if functional:
            return "unknown"
        return None

    except Exception as e:
        print(f"Error checking C++ SSIMULACRA2: {e}")
        return None


def get_rust_ssimulacra2_version():
    """Get the version of the Rust SSIMULACRA2 implementation.

    Returns:
        str: Version string or None if not available
    """
    try:
        # Check if executable exists
        which_result = subprocess.run(
            ["which", "as2c"], capture_output=True, text=True, check=False
        )
        if which_result.returncode != 0:
            return None  # Not installed

        # Try with version flag
        version_result = subprocess.run(
            ["as2c", "--version"], capture_output=True, text=True, check=False
        )

        # Verify functionality
        functional = False
        with tempfile.TemporaryDirectory() as temp_dir:
            test_img1 = Path(temp_dir) / "test1.png"
            test_img2 = Path(temp_dir) / "test2.png"

            # Create white images
            Image.new("RGB", (10, 10), color="white").save(test_img1)
            Image.new("RGB", (10, 10), color="white").save(test_img2)

            # Run test
            test_result = subprocess.run(
                ["as2c", str(test_img1), str(test_img2)],
                capture_output=True,
                text=True,
                check=False,
            )

            if test_result.returncode == 0 and test_result.stdout.strip():
                try:
                    float(test_result.stdout.strip())
                    functional = True
                except ValueError:
                    pass

        if not functional:
            return None  # Not working properly

        # Parse version from --version output
        if version_result.returncode == 0 and version_result.stdout.strip():
            # Check for format: "as2c 0.1.3"
            version_match = re.search(
                r"as2c\s+(\d+\.\d+\.\d+|\d+\.\d+)", version_result.stdout
            )
            if version_match:
                return version_match.group(1)
            else:
                # Try generic version pattern as fallback
                version_match = re.search(
                    r"(\d+\.\d+\.\d+|\d+\.\d+)", version_result.stdout
                )
                if version_match:
                    return version_match.group(1)

        # If we get here, it works but version is unknown
        if functional:
            return "unknown"
        return None

    except Exception as e:
        print(f"Error checking Rust SSIMULACRA2: {e}")
        return None


# Get versions on module import but don't print errors
CPP_SSIMULACRA2_VERSION = get_cpp_ssimulacra2_version()
CPP_SSIMULACRA2_AVAILABLE = CPP_SSIMULACRA2_VERSION is not None

RUST_SSIMULACRA2_VERSION = get_rust_ssimulacra2_version()
RUST_SSIMULACRA2_AVAILABLE = RUST_SSIMULACRA2_VERSION is not None


def run_python_ssimulacra2(original_path, compressed_path):
    """Run the Python implementation of SSIMULACRA2.

    Args:
        original_path (str): Path to original image
        compressed_path (str): Path to compressed/decoded image

    Returns:
        float: SSIMULACRA2 score or None if not available

    Raises:
        RuntimeError: If the Python implementation is not available
    """
    if SSIMULACRA2_AVAILABLE:
        return compute_ssimulacra2_with_alpha(original_path, compressed_path)
    else:
        raise RuntimeError("Python SSIMULACRA2 implementation not available")


def run_cpp_ssimulacra2(original_path, compressed_path):
    """Run the C++ implementation of SSIMULACRA2.

    Args:
        original_path (str): Path to original image
        compressed_path (str): Path to compressed/decoded image

    Returns:
        float: SSIMULACRA2 score or None if not available
    """
    if not CPP_SSIMULACRA2_AVAILABLE:
        print("C++ implementation (ssimulacra2) not available")
        return None

    try:
        cmd = ["ssimulacra2", original_path, compressed_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if result.returncode != 0:
            if result.stderr:
                print(f"C++ stderr: {result.stderr}")
            return None

        # Based on the diagnostic output, we know the score is in stdout
        output = result.stdout.strip()
        if not output:
            return None

        # Try to parse the entire output as a float (which is what your implementation does)
        try:
            return float(output)
        except ValueError:
            # If that doesn't work, try line by line
            lines = output.splitlines()
            for line in lines:
                try:
                    # Try to parse the line as a float
                    return float(line.strip())
                except ValueError:
                    # Try to extract a float from the line
                    match = re.search(r"(-?\d+\.\d+)", line)
                    if match:
                        return float(match.group(1))

        # If we get here, couldn't parse the output
        print(f"Could not parse C++ output: '{output}'")
        return None

    except Exception as e:
        print(f"Error with C++ implementation: {e}")
        return None


def run_rust_ssimulacra2(original_path, compressed_path):
    """Run the Rust implementation (as2c) of SSIMULACRA2.

    Args:
        original_path (str): Path to original image
        compressed_path (str): Path to compressed/decoded image

    Returns:
        float: SSIMULACRA2 score or None if not available
    """
    if not RUST_SSIMULACRA2_AVAILABLE:
        print("Rust implementation (as2c) not available")
        return None

    try:
        cmd = ["as2c", original_path, compressed_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if result.returncode != 0:
            if result.stderr:
                print(f"Rust stderr: {result.stderr}")
            return None

        output = result.stdout.strip()
        if not output:
            return None

        # First try to parse the entire output as a float
        try:
            return float(output)
        except ValueError:
            # If that doesn't work, try line by line
            lines = output.splitlines()
            for line in lines:
                try:
                    # Try to parse the line as a float
                    return float(line.strip())
                except ValueError:
                    # Try to extract a float from the line
                    match = re.search(r"(-?\d+\.\d+)", line)
                    if match:
                        return float(match.group(1))

        # If we get here, couldn't parse the output
        print(f"Could not parse Rust output: '{output}'")
        return None

    except Exception as e:
        print(f"Error with Rust implementation: {e}")
        return None
