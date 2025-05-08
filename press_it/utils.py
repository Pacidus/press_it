"""Utility functions for press_it."""

import os
import shutil
import subprocess
from pathlib import Path


class MissingDependencyError(RuntimeError):
    """Custom exception for missing required dependencies."""


def check_dependencies(required_tools):
    """Verify required system utilities are available.

    Args:
        required_tools (list): List of command-line tools to check for

    Raises:
        MissingDependencyError: If any required tool is missing
    """
    missing = []
    for tool in required_tools:
        if not shutil.which(tool):
            missing.append(tool)

    if missing:
        raise MissingDependencyError(
            f"Missing required tools: {', '.join(missing)}. "
            "Please install them before running this script."
        )


def resize_image(file_path, size):
    """Resize an image to the specified dimensions.

    Args:
        file_path (str): Path to the image file
        size (str): Size in the format "widthxheight"

    Returns:
        None
    """
    subprocess.run(
        [
            "magick",
            file_path,
            "-resize",
            size,
            file_path,
        ],
        check=True,
    )
