"""SSIMULACRA2 engine wrappers for press_it benchmarks."""

import subprocess
import re
import importlib.metadata
import sys

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
        # First try with --version flag
        result = subprocess.run(
            ["ssimulacra2", "--version"], capture_output=True, text=True, check=False
        )

        # If that doesn't work, try without any arguments to get the version header
        if result.returncode != 0 or not result.stdout.strip():
            result = subprocess.run(
                ["ssimulacra2"], capture_output=True, text=True, check=False
            )

        if not result.stdout.strip():
            return None

        # Try to extract version from output (format: "SSIMULACRA 2.1 [AVX3_ZEN4,AVX3,AVX2,SSE4,SSSE3]")
        version_match = re.search(r"SSIMULACRA\s+(\d+\.\d+)", result.stdout)
        if version_match:
            return version_match.group(1)
        else:
            # Try generic version pattern as fallback
            version_match = re.search(r"(\d+\.\d+\.\d+|\d+\.\d+)", result.stdout)
            if version_match:
                return version_match.group(1)
            return "unknown"
    except FileNotFoundError:
        return None


def get_rust_ssimulacra2_version():
    """Get the version of the Rust SSIMULACRA2 implementation.

    Returns:
        str: Version string or None if not available
    """
    try:
        result = subprocess.run(
            ["as2c", "--version"], capture_output=True, text=True, check=False
        )

        if result.returncode != 0:
            return None

        # Try to extract version from output
        # Check for format: "as2c 0.1.3"
        version_match = re.search(r"as2c\s+(\d+\.\d+\.\d+|\d+\.\d+)", result.stdout)
        if version_match:
            return version_match.group(1)
        else:
            # Try generic version pattern as fallback
            version_match = re.search(r"(\d+\.\d+\.\d+|\d+\.\d+)", result.stdout)
            if version_match:
                return version_match.group(1)
            return "unknown"
    except FileNotFoundError:
        return None


# Get versions on module import
try:
    CPP_SSIMULACRA2_VERSION = get_cpp_ssimulacra2_version()
    CPP_SSIMULACRA2_AVAILABLE = CPP_SSIMULACRA2_VERSION is not None
except Exception:
    CPP_SSIMULACRA2_VERSION = None
    CPP_SSIMULACRA2_AVAILABLE = False

try:
    RUST_SSIMULACRA2_VERSION = get_rust_ssimulacra2_version()
    RUST_SSIMULACRA2_AVAILABLE = RUST_SSIMULACRA2_VERSION is not None
except Exception:
    RUST_SSIMULACRA2_VERSION = None
    RUST_SSIMULACRA2_AVAILABLE = False


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

        output = result.stdout.strip()
        if not output:
            return None

        try:
            # Try to extract just the score (first line or first number)
            # Some versions might output additional information
            lines = output.splitlines()
            for line in lines:
                try:
                    # Try to parse the line as a float
                    return float(line.strip())
                except ValueError:
                    # If line contains a number with other text, extract the first number
                    number_match = re.search(r"(-?\d+\.\d+)", line)
                    if number_match:
                        return float(number_match.group(1))

            # If we get here, couldn't parse any line as a float
            return None

        except ValueError:
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

        try:
            # Try to extract just the score (first line or first number)
            lines = output.splitlines()
            for line in lines:
                try:
                    # Try to parse the line as a float
                    return float(line.strip())
                except ValueError:
                    # If line contains a number with other text, extract the first number
                    number_match = re.search(r"(-?\d+\.\d+)", line)
                    if number_match:
                        return float(number_match.group(1))

            # If we get here, couldn't parse any line as a float
            return None

        except ValueError:
            return None

    except Exception as e:
        print(f"Error with Rust implementation: {e}")
        return None
