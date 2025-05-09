"""SSIMULACRA2 engine wrappers for press_it benchmarks."""

import re
import functools
import tempfile
from pathlib import Path

from press_it.utils.subprocess_utils import run_command, check_command_exists


def try_import(module_name):
    """Try to import a module, return None if not available."""
    try:
        module = __import__(module_name)
        return module
    except ImportError:
        return None


# Try to import the Python implementation of SSIMULACRA2
ssimulacra2 = try_import("ssimulacra2")
PYTHON_SSIMULACRA2_AVAILABLE = ssimulacra2 is not None


def get_version(executable, version_args=None, patterns=None, test_function=None):
    """Generic function to get the version of a command-line tool.

    Args:
        executable: Name of the executable
        version_args: Arguments to get version (default: --version)
        patterns: List of regex patterns to extract version
        test_function: Function to test functionality

    Returns:
        str: Version string or None if not available
    """
    # Check if executable exists
    if not check_command_exists(executable):
        return None

    # Set defaults
    if version_args is None:
        version_args = ["--version"]

    if patterns is None:
        patterns = [
            r"(\d+\.\d+\.\d+)",  # Match x.y.z
            r"(\d+\.\d+)",  # Match x.y
            r"v?(\d+\.\d+)",  # Match vx.y
        ]

    # Try to get version
    try:
        result = run_command([executable] + version_args, check=False, silent=True)

        # Check both stdout and stderr for version
        output = result.stdout + "\n" + result.stderr

        # Try each pattern
        for pattern in patterns:
            match = re.search(pattern, output)
            if match:
                # If test function provided, verify functionality
                if test_function and not test_function(executable):
                    return None

                return match.group(1)

        # If no match but command exists
        if test_function and test_function(executable):
            return "unknown"

    except Exception:
        pass

    return None


def create_test_images():
    """Create test images for functionality testing.

    Returns:
        tuple: (path1, path2) - Paths to two test images
    """
    from PIL import Image

    temp_dir = tempfile.TemporaryDirectory()

    # Create white test images
    path1 = Path(temp_dir.name) / "test1.png"
    path2 = Path(temp_dir.name) / "test2.png"

    Image.new("RGB", (10, 10), color="white").save(path1)
    Image.new("RGB", (10, 10), color="white").save(path2)

    # Return both the paths and the temp dir to prevent cleanup
    return path1, path2, temp_dir


def test_cpp_ssimulacra2(executable="ssimulacra2"):
    """Test if C++ SSIMULACRA2 is functional.

    Args:
        executable: Name of the executable

    Returns:
        bool: True if functional
    """
    try:
        # Create test images
        path1, path2, temp_dir = create_test_images()

        # Run test
        result = run_command(
            [executable, str(path1), str(path2)], check=False, silent=True
        )

        # Check if it produced a valid score
        if result.returncode == 0 and result.stdout.strip():
            try:
                float(result.stdout.strip())
                return True
            except ValueError:
                pass

        return False

    except Exception:
        return False
    finally:
        # Clean up
        if "temp_dir" in locals():
            temp_dir.cleanup()


def test_rust_ssimulacra2(executable="as2c"):
    """Test if Rust SSIMULACRA2 is functional.

    Args:
        executable: Name of the executable

    Returns:
        bool: True if functional
    """
    try:
        # Create test images
        path1, path2, temp_dir = create_test_images()

        # Run test
        result = run_command(
            [executable, str(path1), str(path2)], check=False, silent=True
        )

        # Check if it produced a valid score
        if result.returncode == 0 and result.stdout.strip():
            try:
                float(result.stdout.strip())
                return True
            except ValueError:
                pass

        return False

    except Exception:
        return False
    finally:
        # Clean up
        if "temp_dir" in locals():
            temp_dir.cleanup()


# Get versions of different implementations
PYTHON_SSIMULACRA2_VERSION = (
    getattr(ssimulacra2, "__version__", None) if ssimulacra2 else None
)

CPP_SSIMULACRA2_VERSION = get_version(
    "ssimulacra2",
    version_args=[],  # No arguments, version is in stderr
    patterns=[r"SSIMULACRA\s+(\d+\.\d+)"],
    test_function=test_cpp_ssimulacra2,
)
CPP_SSIMULACRA2_AVAILABLE = CPP_SSIMULACRA2_VERSION is not None

RUST_SSIMULACRA2_VERSION = get_version(
    "as2c",
    version_args=["--version"],
    patterns=[r"as2c\s+(\d+\.\d+\.\d+|\d+\.\d+)"],
    test_function=test_rust_ssimulacra2,
)
RUST_SSIMULACRA2_AVAILABLE = RUST_SSIMULACRA2_VERSION is not None


def cached_result(func):
    """Decorator to cache function results based on input paths."""
    cache = {}

    @functools.wraps(func)
    def wrapper(input_path, decoded_path):
        key = (str(input_path), str(decoded_path))
        if key in cache:
            return cache[key]

        result = func(input_path, decoded_path)
        cache[key] = result
        return result

    return wrapper


@cached_result
def run_python_ssimulacra2(input_path, decoded_path):
    """Run the Python implementation of SSIMULACRA2.

    Args:
        input_path: Path to original image
        decoded_path: Path to compressed/decoded image

    Returns:
        float: SSIMULACRA2 score or None if not available
    """
    if not PYTHON_SSIMULACRA2_AVAILABLE:
        return None

    try:
        return ssimulacra2.compute_ssimulacra2_with_alpha(input_path, decoded_path)
    except Exception as e:
        print(f"Error running Python SSIMULACRA2: {e}")
        return None


@cached_result
def run_cpp_ssimulacra2(input_path, decoded_path):
    """Run the C++ implementation of SSIMULACRA2.

    Args:
        input_path: Path to original image
        decoded_path: Path to compressed/decoded image

    Returns:
        float: SSIMULACRA2 score or None if not available
    """
    if not CPP_SSIMULACRA2_AVAILABLE:
        return None

    try:
        result = run_command(
            ["ssimulacra2", str(input_path), str(decoded_path)], check=True, silent=True
        )

        # Try to parse the output as a float
        try:
            return float(result.stdout.strip())
        except ValueError:
            # Try to extract a float from the output
            match = re.search(r"(-?\d+\.\d+)", result.stdout)
            if match:
                return float(match.group(1))

        return None

    except Exception as e:
        print(f"Error running C++ SSIMULACRA2: {e}")
        return None


@cached_result
def run_rust_ssimulacra2(input_path, decoded_path):
    """Run the Rust implementation (as2c) of SSIMULACRA2.

    Args:
        input_path: Path to original image
        decoded_path: Path to compressed/decoded image

    Returns:
        float: SSIMULACRA2 score or None if not available
    """
    if not RUST_SSIMULACRA2_AVAILABLE:
        return None

    try:
        result = run_command(
            ["as2c", str(input_path), str(decoded_path)], check=True, silent=True
        )

        # Try to parse the output as a float
        try:
            return float(result.stdout.strip())
        except ValueError:
            # Try to extract a float from the output
            match = re.search(r"(-?\d+\.\d+)", result.stdout)
            if match:
                return float(match.group(1))

        return None

    except Exception as e:
        print(f"Error running Rust SSIMULACRA2: {e}")
        return None


def get_best_ssimulacra2():
    """Get the best available SSIMULACRA2 implementation.

    Returns:
        function: The best available implementation
    """
    # Priority: Python (most likely available), C++, Rust
    if PYTHON_SSIMULACRA2_AVAILABLE:
        return run_python_ssimulacra2
    elif CPP_SSIMULACRA2_AVAILABLE:
        return run_cpp_ssimulacra2
    elif RUST_SSIMULACRA2_AVAILABLE:
        return run_rust_ssimulacra2
    else:
        raise RuntimeError("No SSIMULACRA2 implementation available")


def run_all_implementations(input_path, decoded_path):
    """Run all available SSIMULACRA2 implementations.

    Args:
        input_path: Path to original image
        decoded_path: Path to compressed/decoded image

    Returns:
        dict: Results from all implementations
    """
    results = {}

    # Run Python implementation
    python_score = run_python_ssimulacra2(input_path, decoded_path)
    if python_score is not None:
        results["python"] = python_score

    # Run C++ implementation
    cpp_score = run_cpp_ssimulacra2(input_path, decoded_path)
    if cpp_score is not None:
        results["cpp"] = cpp_score

    # Run Rust implementation
    rust_score = run_rust_ssimulacra2(input_path, decoded_path)
    if rust_score is not None:
        results["rust"] = rust_score

    return results
