"""Report generation for benchmark analysis."""

import os
from pathlib import Path
import json

from press_it.benchmark.analysis.summary import summarize_benchmark_data
from press_it.benchmark.analysis.consistency import (
    analyze_encoder_consistency,
    analyze_implementation_bias,
    analyze_implementation_reliability,
)
from press_it.benchmark.analysis.efficiency import (
    analyze_image_factors,
    analyze_compression_vs_format,
)
from press_it.benchmark.analysis.quality import (
    analyze_quality_distribution,
    analyze_quality_thresholds,
)


def generate_benchmark_report(df, output_path):
    """Generate a comprehensive benchmark report.

    Args:
        df: DataFrame with benchmark data
        output_path: Path to save the report

    Returns:
        str: Path to the generated report
    """
    # Generate analysis data
    summary = summarize_benchmark_data(df)
    consistency = analyze_encoder_consistency(df)
    image_factors = analyze_image_factors(df)
    format_comparison = analyze_compression_vs_format(df)
    quality_thresholds = analyze_quality_thresholds(df)
    implementation_bias = analyze_implementation_bias(df)
    implementation_reliability = analyze_implementation_reliability(df)

    # Create report
    with open(output_path, "w") as f:
        write_report_header(f)
        write_summary_section(f, summary)
        write_format_comparison_section(f, format_comparison, summary)
        write_quality_section(f, quality_thresholds, summary)
        write_image_factors_section(f, image_factors)
        write_implementation_section(
            f, consistency, implementation_bias, implementation_reliability
        )
        write_conclusion_section(f, summary, format_comparison, quality_thresholds)

    return output_path


def write_report_header(file):
    """Write report header section.

    Args:
        file: File object to write to
    """
    file.write("# SSIMULACRA2 Benchmark Analysis Report\n\n")
    file.write("This report analyzes the performance of different image compression ")
    file.write("formats using SSIMULACRA2 perceptual quality metrics.\n\n")


def write_summary_section(file, summary):
    """Write summary section of the report.

    Args:
        file: File object to write to
        summary: Summary statistics dictionary
    """
    file.write("## Summary\n\n")

    # Basic info
    file.write(f"- Total samples: {summary['total_samples']}\n")

    # Compression type distribution
    file.write("\n### Compression Type Distribution\n\n")
    for comp_type, stats in summary["compression_types"].items():
        file.write(
            f"- {comp_type}: {stats['count']} samples ({stats['percentage']:.2f}%)\n"
        )

    # Add image dimensions if available
    if "image_dimensions" in summary and summary["image_dimensions"]:
        file.write("\n### Image Dimensions\n\n")
        width_stats = summary["image_dimensions"]["width"]
        height_stats = summary["image_dimensions"]["height"]
        file.write(f"- Width: {width_stats['min']} to {width_stats['max']} pixels ")
        file.write(f"(avg: {width_stats['mean']})\n")
        file.write(f"- Height: {height_stats['min']} to {height_stats['max']} pixels ")
        file.write(f"(avg: {height_stats['mean']})\n")

    # Quality scores
    if "score_stats" in summary and "overall" in summary["score_stats"]:
        file.write("\n### Quality Score Statistics\n\n")
        file.write(
            f"- Overall: min={summary['score_stats']['overall']['min']:.2f}, "
            f"max={summary['score_stats']['overall']['max']:.2f}, "
            f"mean={summary['score_stats']['overall']['mean']:.2f}, "
            f"median={summary['score_stats']['overall']['median']:.2f}\n"
        )

        if "by_compression_type" in summary["score_stats"]:
            file.write("\nBy format:\n\n")
            for comp_type, stats in summary["score_stats"][
                "by_compression_type"
            ].items():
                file.write(
                    f"- {comp_type}: min={stats['min']:.2f}, "
                    f"max={stats['max']:.2f}, "
                    f"mean={stats['mean']:.2f}, "
                    f"median={stats['median']:.2f}\n"
                )


def write_format_comparison_section(file, format_comparison, summary):
    """Write format comparison section.

    Args:
        file: File object to write to
        format_comparison: Format comparison results
        summary: Summary statistics dictionary
    """
    file.write("\n## Compression Format Comparison\n\n")

    # Compression statistics
    if "compression_stats" in summary and "overall" in summary["compression_stats"]:
        file.write("### Compression Ratio Statistics\n\n")
        file.write(
            f"- Overall: min={summary['compression_stats']['overall']['min']:.2f}x, "
            f"max={summary['compression_stats']['overall']['max']:.2f}x, "
            f"mean={summary['compression_stats']['overall']['mean']:.2f}x, "
            f"median={summary['compression_stats']['overall']['median']:.2f}x\n\n"
        )

        if "by_compression_type" in summary["compression_stats"]:
            file.write("By format:\n\n")

            # Create a table for better readability
            file.write("| Format | Min | Max | Mean | Median |\n")
            file.write("|--------|-----|-----|------|--------|\n")

            for comp_type, stats in summary["compression_stats"][
                "by_compression_type"
            ].items():
                file.write(
                    f"| {comp_type} | {stats['min']:.2f}x | {stats['max']:.2f}x | "
                    f"{stats['mean']:.2f}x | {stats['median']:.2f}x |\n"
                )

    # Format efficiency rankings
    if format_comparison:
        file.write("\n### Format Efficiency Rankings\n\n")

        # Sort formats by average compression ratio
        sorted_formats = sorted(
            format_comparison.keys(),
            key=lambda x: format_comparison[x]["mean_compression"],
            reverse=True,
        )

        file.write("| Rank | Format | Avg. Compression | Samples |\n")
        file.write("|------|--------|------------------|--------|\n")

        for i, fmt in enumerate(sorted_formats):
            stats = format_comparison[fmt]
            file.write(
                f"| {i+1} | {fmt} | {stats['mean_compression']:.2f}x | {stats['samples']} |\n"
            )

    # Most efficient formats by quality level
    if "most_efficient_format" in summary:
        file.write("\n### Most Efficient Format by Quality Level\n\n")

        file.write("| Quality Range | Best Format |\n")
        file.write("|---------------|------------|\n")

        for quality_range, format_name in summary["most_efficient_format"].items():
            file.write(f"| {quality_range} | **{format_name}** |\n")


def write_quality_section(file, quality_thresholds, summary):
    """Write quality analysis section.

    Args:
        file: File object to write to
        quality_thresholds: Quality threshold analysis
        summary: Summary statistics dictionary
    """
    file.write("\n## Quality Analysis\n\n")

    # Quality thresholds
    if quality_thresholds:
        file.write("### Minimum Quality Settings for Target Scores\n\n")
        file.write(
            "This table shows the minimum quality setting required to achieve specific quality targets:\n\n"
        )

        # Get available thresholds and formats
        thresholds = list(
            set(
                threshold
                for fmt_data in quality_thresholds.values()
                for threshold in fmt_data.keys()
            )
        )
        formats = list(quality_thresholds.keys())

        # Sort thresholds numerically
        thresholds.sort(key=lambda x: float(x))

        # Create table header
        file.write("| Format |")
        for threshold in thresholds:
            file.write(f" Quality {threshold}+ |")
        file.write("\n")

        # Create separator row
        file.write("|--------|")
        for _ in thresholds:
            file.write("----------|")
        file.write("\n")

        # Add data rows
        for fmt in formats:
            file.write(f"| {fmt} |")
            for threshold in thresholds:
                if threshold in quality_thresholds[fmt]:
                    data = quality_thresholds[fmt][threshold]
                    file.write(
                        f" Q={data['min_quality_setting']} ({data['avg_compression_ratio']:.2f}x) |"
                    )
                else:
                    file.write(" N/A |")
            file.write("\n")

    # Implementation consistency
    file.write("\n### Quality Consistency\n\n")

    # Add implementation availability info
    if "implementation_availability" in summary:
        file.write("SSIMULACRA2 implementation availability in test data:\n\n")
        for impl, percentage in summary["implementation_availability"].items():
            file.write(f"- {impl}: {percentage:.2f}% of samples\n")


def write_image_factors_section(file, image_factors):
    """Write image factors section.

    Args:
        file: File object to write to
        image_factors: Image factor analysis results
    """
    file.write("\n## Image Size Impact\n\n")

    if image_factors and "image_size_impact" in image_factors:
        file.write(
            "This analysis shows how image size affects compression performance:\n\n"
        )

        # Create a table for better readability
        file.write("| Image Size | Best Format | Avg. Compression | Samples |\n")
        file.write("|------------|-------------|-------------------|--------|\n")

        for size_cat, stats in image_factors["image_size_impact"].items():
            file.write(
                f"| {size_cat} | {stats['best_format']} | "
                f"{stats['avg_compression']:.2f}x | {stats['samples']} |\n"
            )
    else:
        file.write("Insufficient data to analyze image size impact.\n")


def write_implementation_section(
    file, consistency, implementation_bias, implementation_reliability
):
    """Write implementation analysis section.

    Args:
        file: File object to write to
        consistency: Consistency analysis results
        implementation_bias: Implementation bias analysis results
        implementation_reliability: Implementation reliability analysis results
    """
    file.write("\n## SSIMULACRA2 Implementation Analysis\n\n")

    # Implementation consistency
    if consistency:
        file.write("### Implementation Consistency\n\n")
        file.write(
            "This section shows how consistently different SSIMULACRA2 implementations score the same images:\n\n"
        )

        # Create a table for better readability
        file.write(
            "| Comparison | Samples | Correlation | Mean Difference | Max Difference |\n"
        )
        file.write(
            "|------------|---------|-------------|-----------------|---------------|\n"
        )

        for comparison, stats in consistency.items():
            file.write(
                f"| {comparison} | {stats['samples']} | {stats['correlation']:.3f} | "
                f"{stats['mean_difference']:.3f} | {stats['max_difference']:.3f} |\n"
            )

    # Implementation bias
    if implementation_bias and "overall" in implementation_bias:
        file.write("\n### Implementation Bias\n\n")
        file.write(
            "This analysis shows whether different implementations tend to score higher or lower:\n\n"
        )

        overall = implementation_bias["overall"]
        file.write(
            f"- Overall: {overall['highest_impl']} scores highest (avg: {overall['scores'][overall['highest_impl']]:.2f}), "
        )
        file.write(
            f"{overall['lowest_impl']} scores lowest (avg: {overall['scores'][overall['lowest_impl']]:.2f})\n"
        )
        file.write(
            f"- Max difference: {overall['max_difference']:.2f} points ({overall['relative_difference']:.2f}%)\n"
        )

    # Implementation reliability
    if implementation_reliability:
        file.write("\n### Implementation Reliability\n\n")
        file.write(
            "This analysis shows how reliable (consistent) each implementation is:\n\n"
        )

        # Create a table for better readability
        file.write("| Implementation | Availability | Avg. CV | Reliability Rank |\n")
        file.write("|----------------|-------------|---------|------------------|\n")

        # Sort by reliability rank if available
        sorted_impls = sorted(
            [
                impl
                for impl in implementation_reliability.keys()
                if "reliability_rank" in implementation_reliability[impl]
            ],
            key=lambda x: implementation_reliability[x]["reliability_rank"],
        )

        # Add other implementations without reliability rank
        sorted_impls.extend(
            [
                impl
                for impl in implementation_reliability.keys()
                if "reliability_rank" not in implementation_reliability[impl]
            ]
        )

        for impl in sorted_impls:
            stats = implementation_reliability[impl]
            rank_str = (
                str(stats["reliability_rank"]) if "reliability_rank" in stats else "N/A"
            )
            cv_str = f"{stats['avg_cv']:.4f}" if "avg_cv" in stats else "N/A"

            file.write(
                f"| {impl} | {stats['availability']:.1f}% | {cv_str} | {rank_str} |\n"
            )


def write_conclusion_section(file, summary, format_comparison, quality_thresholds):
    """Write conclusion section.

    Args:
        file: File object to write to
        summary: Summary statistics
        format_comparison: Format comparison results
        quality_thresholds: Quality threshold analysis
    """
    file.write("\n## Conclusions\n\n")

    # Find overall best format
    best_format = None
    best_ratio = 0

    if (
        "compression_stats" in summary
        and "by_compression_type" in summary["compression_stats"]
    ):
        for comp_type, stats in summary["compression_stats"][
            "by_compression_type"
        ].items():
            if stats["mean"] > best_ratio:
                best_ratio = stats["mean"]
                best_format = comp_type

    # Write overall best format
    if best_format:
        file.write(
            f"- Best overall format: **{best_format}** with average "
            f"compression ratio of {best_ratio:.2f}x\n"
        )

    # Write format recommendations by quality level
    if "most_efficient_format" in summary and summary["most_efficient_format"]:
        file.write("\n### Format Recommendations by Quality Level\n\n")
        for quality_range, format_name in summary["most_efficient_format"].items():
            file.write(f"- For quality {quality_range}: Use **{format_name}**\n")

    # Write practical quality settings if available
    if quality_thresholds:
        file.write("\n### Recommended Quality Settings\n\n")
        file.write("To achieve good quality scores while maximizing compression:\n\n")

        # Get formats and focus on 80+ quality threshold
        target_threshold = "80"
        formats_with_threshold = [
            fmt
            for fmt in quality_thresholds.keys()
            if target_threshold in quality_thresholds[fmt]
        ]

        if formats_with_threshold:
            for fmt in formats_with_threshold:
                data = quality_thresholds[fmt][target_threshold]
                file.write(
                    f"- For {fmt}: Use quality setting **{data['min_quality_setting']}** "
                    f"(achieves {target_threshold}+ quality score with {data['avg_compression_ratio']:.2f}x compression)\n"
                )

    file.write("\n*Generated by press_it benchmark analysis*\n")


def generate_json_report(df, output_path):
    """Generate a JSON format report with all analysis data.

    Args:
        df: DataFrame with benchmark data
        output_path: Path to save the report

    Returns:
        str: Path to the generated report
    """
    # Generate analysis data
    results = {
        "summary": summarize_benchmark_data(df),
        "consistency": analyze_encoder_consistency(df),
        "implementation_bias": analyze_implementation_bias(df),
        "implementation_reliability": analyze_implementation_reliability(df),
        "image_factors": analyze_image_factors(df),
        "format_comparison": analyze_compression_vs_format(df),
        "quality_thresholds": analyze_quality_thresholds(df),
        "quality_distribution": analyze_quality_distribution(df),
        "metadata": {
            "num_samples": len(df),
            "timestamp": pd.Timestamp.now().isoformat(),
        },
    }

    # Save to JSON
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    return output_path
