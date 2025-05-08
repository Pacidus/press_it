"""SSIMULACRA2 engine wrappers for press_it benchmarks."""

import subprocess

# Try to import the Python implementation of SSIMULACRA2
try:
    from ssimulacra2 import compute_ssimulacra2_with_alpha

    SSIMULACRA2_AVAILABLE = True
except ImportError:
    print(
        "Warning: Could not import ssimulacra2 module. Python implementation will be skipped."
    )
    SSIMULACRA2_AVAILABLE = False


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

    Raises:
        subprocess.CalledProcessError: If the command fails
        ValueError: If the output cannot be parsed
    """
    try:
        cmd = ["cmulacra2", original_path, compressed_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        print(f"Error running C++ implementation: {e}")
        print(f"stderr: {e.stderr}")
        return None
    except FileNotFoundError:
        print("C++ implementation (cmulacra2) not found in PATH")
        return None
    except Exception as e:
        print(f"Unexpected error with C++ implementation: {e}")
        return None


def run_rust_ssimulacra2(original_path, compressed_path):
    """Run the Rust implementation (as2c) of SSIMULACRA2.

    Args:
        original_path (str): Path to original image
        compressed_path (str): Path to compressed/decoded image

    Returns:
        float: SSIMULACRA2 score or None if not available

    Raises:
        subprocess.CalledProcessError: If the command fails
        ValueError: If the output cannot be parsed
    """
    try:
        cmd = ["as2c", original_path, compressed_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        print(f"Error running Rust implementation: {e}")
        print(f"stderr: {e.stderr}")
        return None
    except FileNotFoundError:
        print("Rust implementation (as2c) not found in PATH")
        return None
    except Exception as e:
        print(f"Unexpected error with Rust implementation: {e}")
        return None
