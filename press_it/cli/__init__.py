"""Command-line interfaces for press_it."""

from press_it.cli.compression_cli import main as compression_main
from press_it.cli.benchmark_cli import main as benchmark_main

# Define what's available when doing "from press_it.cli import *"
__all__ = ["compression_main", "benchmark_main"]
