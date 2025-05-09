"""Benchmark module for press_it."""

from press_it.benchmark.engines import (
    PYTHON_SSIMULACRA2_AVAILABLE,
    CPP_SSIMULACRA2_AVAILABLE,
    RUST_SSIMULACRA2_AVAILABLE,
    PYTHON_SSIMULACRA2_VERSION,
    CPP_SSIMULACRA2_VERSION,
    RUST_SSIMULACRA2_VERSION,
    run_python_ssimulacra2,
    run_cpp_ssimulacra2,
    run_rust_ssimulacra2,
    get_best_ssimulacra2,
    run_all_implementations,
)

from press_it.benchmark.image import (
    download_image,
    get_random_image,
    encode_with_format,
    encode_random_format,
)

from press_it.benchmark.runner import BenchmarkRunner

from press_it.benchmark.analysis import (
    find_benchmark_files,
    load_benchmark_data,
    combine_benchmark_files,
    summarize_benchmark_data,
    print_summary,
    analyze_quality_vs_size,
    analyze_efficiency_by_quality,
    analyze_encoder_consistency,
    analyze_image_factors,
    generate_benchmark_report,
)

# Define what's available when doing "from press_it.benchmark import *"
__all__ = [
    # SSIMULACRA2 implementations
    "PYTHON_SSIMULACRA2_AVAILABLE",
    "CPP_SSIMULACRA2_AVAILABLE",
    "RUST_SSIMULACRA2_AVAILABLE",
    "PYTHON_SSIMULACRA2_VERSION",
    "CPP_SSIMULACRA2_VERSION",
    "RUST_SSIMULACRA2_VERSION",
    "run_python_ssimulacra2",
    "run_cpp_ssimulacra2",
    "run_rust_ssimulacra2",
    "get_best_ssimulacra2",
    "run_all_implementations",
    # Image handling
    "download_image",
    "get_random_image",
    "encode_with_format",
    "encode_random_format",
    # Benchmark runner
    "BenchmarkRunner",
    # Analysis
    "find_benchmark_files",
    "load_benchmark_data",
    "combine_benchmark_files",
    "summarize_benchmark_data",
    "print_summary",
    "analyze_quality_vs_size",
    "analyze_efficiency_by_quality",
    "analyze_encoder_consistency",
    "analyze_image_factors",
    "generate_benchmark_report",
]
