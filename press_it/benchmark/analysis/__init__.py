"""Analysis utilities for press_it benchmark data."""

# Import core functionality
from press_it.benchmark.analysis.core import (
    find_benchmark_files,
    load_benchmark_data,
    combine_benchmark_files,
    add_categorical_columns,
)

# Import summary statistics
from press_it.benchmark.analysis.summary import summarize_benchmark_data, print_summary

# Import efficiency analysis
from press_it.benchmark.analysis.efficiency import (
    analyze_quality_vs_size,
    analyze_efficiency_by_quality,
    analyze_image_factors,
)

# Import quality analysis
from press_it.benchmark.analysis.quality import analyze_quality_distribution

# Import consistency analysis
from press_it.benchmark.analysis.consistency import analyze_encoder_consistency

# Import reporting
from press_it.benchmark.analysis.reporting import generate_benchmark_report

# Define what's available when doing "from press_it.benchmark.analysis import *"
__all__ = [
    # Core data handling
    "find_benchmark_files",
    "load_benchmark_data",
    "combine_benchmark_files",
    "add_categorical_columns",
    # Summary statistics
    "summarize_benchmark_data",
    "print_summary",
    # Efficiency analysis
    "analyze_quality_vs_size",
    "analyze_efficiency_by_quality",
    "analyze_image_factors",
    # Quality analysis
    "analyze_quality_distribution",
    # Consistency analysis
    "analyze_encoder_consistency",
    # Reporting
    "generate_benchmark_report",
]
