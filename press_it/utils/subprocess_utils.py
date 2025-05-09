"""Utilities for subprocess execution with error handling."""

import subprocess
import os
import functools


def run_command(cmd, check=True, capture_output=True, silent=False):
    """Run a subprocess command with consistent handling.

    Args:
        cmd: Command to run as a list of strings
        check: Whether to raise an exception if the command fails
        capture_output: Whether to capture stdout/stderr
        silent: Whether to suppress printing command output

    Returns:
        CompletedProcess instance
    """
    if not silent:
        print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            check=False,  # We'll handle errors ourselves
            capture_output=capture_output,
            text=True,
        )

        if check and result.returncode != 0:
            error_msg = result.stderr if capture_output else "Unknown error"
            print(f"Command failed: {' '.join(cmd)}")
            print(f"Error: {error_msg}")
            raise subprocess.CalledProcessError(
                result.returncode, cmd, result.stdout, result.stderr
            )

        return result

    except FileNotFoundError as e:
        print(f"Command not found: {cmd[0]}")
        if check:
            raise
        return None


def with_error_handling(func):
    """Decorator to add error handling to subprocess functions."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except subprocess.CalledProcessError as e:
            print(f"Error in {func.__name__}: {e}")
            # Re-raise with a more informative message
            raise RuntimeError(f"Failed to execute {func.__name__}: {e}") from e
        except FileNotFoundError as e:
            print(f"Required command not found: {e}")
            raise RuntimeError(f"Missing required command: {e}") from e

    return wrapper


def check_command_exists(command):
    """Check if a command exists in the system PATH.

    Args:
        command: Command name to check

    Returns:
        bool: True if command exists, False otherwise
    """
    try:
        # Use 'which' on Unix-like systems
        run_command(["which", command], silent=True)
        return True
    except subprocess.CalledProcessError:
        return False


def check_dependencies(required_tools):
    """Verify required system utilities are available.

    Args:
        required_tools: List of command-line tools to check for

    Raises:
        RuntimeError: If any required tool is missing
    """
    missing = []
    for tool in required_tools:
        if not check_command_exists(tool):
            missing.append(tool)

    if missing:
        raise RuntimeError(
            f"Missing required tools: {', '.join(missing)}. "
            "Please install them before running this script."
        )
