"""Visualization utilities for benchmark analysis."""

# These functions are optional and only run if matplotlib is available
# Try to import matplotlib and seaborn
try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    MATPLOTLIB_AVAILABLE = True

    try:
        import seaborn as sns

        SEABORN_AVAILABLE = True
    except ImportError:
        SEABORN_AVAILABLE = False

except ImportError:
    MATPLOTLIB_AVAILABLE = False
    SEABORN_AVAILABLE = False

import pandas as pd
import numpy as np

from press_it.benchmark.analysis.core import (
    add_categorical_columns,
    get_best_score_column,
)
from press_it.benchmark.analysis.efficiency import analyze_quality_vs_size


def check_visualization_available():
    """Check if visualization is available.

    Returns:
        bool: True if matplotlib is available, False otherwise
    """
    return MATPLOTLIB_AVAILABLE


def plot_quality_vs_compression(df, output_path=None):
    """Create a plot of quality scores vs compression ratio.

    Args:
        df: DataFrame with benchmark data
        output_path: Path to save the plot image

    Returns:
        bool: True if successful, False otherwise
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib is not available. Cannot create plot.")
        return False

    if len(df) == 0:
        print("No data to plot")
        return False

    # Get analysis data
    analysis_df = analyze_quality_vs_size(df)

    if analysis_df is None or len(analysis_df) == 0:
        print("No valid analysis data to plot")
        return False

    # Create plot
    plt.figure(figsize=(12, 8))

    # Set style if seaborn is available
    if SEABORN_AVAILABLE:
        sns.set_style("whitegrid")
        palette = sns.color_palette(
            "viridis", n_colors=len(analysis_df["compression_type"].unique())
        )
    else:
        # Default colors if seaborn not available
        palette = plt.cm.tab10(
            np.linspace(0, 1, len(analysis_df["compression_type"].unique()))
        )

    # Plot data by compression type
    for i, (comp_type, group) in enumerate(analysis_df.groupby("compression_type")):
        color = palette[i]
        plt.scatter(
            group["best_score"],
            group["compression_ratio"],
            label=comp_type,
            color=color,
            alpha=0.7,
            s=50,
        )

        # Add smoothed line if there are enough points
        if len(group) >= 3:
            if SEABORN_AVAILABLE:
                # Use LOWESS smoother with seaborn
                sns.regplot(
                    x="best_score",
                    y="compression_ratio",
                    data=group,
                    scatter=False,
                    lowess=True,
                    line_kws={"color": color, "lw": 2},
                    ax=plt.gca(),
                )
            else:
                # Simple polynomial fit without seaborn
                z = np.polyfit(group["best_score"], group["compression_ratio"], 2)
                p = np.poly1d(z)

                # Create smooth x points for the curve
                x_smooth = np.linspace(
                    group["best_score"].min(), group["best_score"].max(), 100
                )
                plt.plot(x_smooth, p(x_smooth), color=color, lw=2)

    # Set labels and title
    plt.xlabel("Perceptual Quality Score (SSIMULACRA2)", fontsize=12)
    plt.ylabel("Compression Ratio", fontsize=12)
    plt.title("Quality vs. Compression Ratio by Format", fontsize=14)

    # Format y-axis to show compression ratio as "Nx"
    plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}x"))

    # Add grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)

    # Add reference lines
    plt.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)  # No compression

    # Add common quality thresholds
    for quality in [60, 70, 80, 90]:
        plt.axvline(x=quality, color="gray", linestyle=":", alpha=0.3)
        plt.text(
            quality, plt.ylim()[0], f"{quality}", ha="center", va="bottom", alpha=0.7
        )

    # Save or show
    if output_path:
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved quality vs. compression plot to {output_path}")
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

    return True


def plot_format_comparison(df, output_path=None):
    """Create a heatmap comparing formats at different quality levels.

    Args:
        df: DataFrame with benchmark data
        output_path: Path to save the plot image

    Returns:
        bool: True if successful, False otherwise
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib is not available. Cannot create plot.")
        return False

    if len(df) == 0:
        print("No data to plot")
        return False

    # Ensure we have quality bins
    analysis_data = add_categorical_columns(df)

    if "quality_bin" not in analysis_data.columns:
        print("Cannot create quality bins for analysis")
        return False

    # Calculate average compression ratio by format and quality bin
    pivot_data = (
        analysis_data.groupby(["compression_type", "quality_bin"])["compression_ratio"]
        .mean()
        .reset_index()
        .pivot(
            index="quality_bin", columns="compression_type", values="compression_ratio"
        )
    )

    if len(pivot_data) == 0:
        print("No valid pivot data to plot")
        return False

    # Create plot
    plt.figure(figsize=(10, 6))

    if SEABORN_AVAILABLE:
        # Create heatmap with seaborn
        ax = sns.heatmap(
            pivot_data,
            annot=True,
            fmt=".1f",
            cmap="viridis",
            linewidths=0.5,
            cbar_kws={"label": "Compression Ratio (higher is better)"},
        )

        # Highlight the best format in each quality range
        for i, quality_range in enumerate(pivot_data.index):
            best_format = pivot_data.loc[quality_range].idxmax()
            best_value = pivot_data.loc[quality_range, best_format]
            j = list(pivot_data.columns).index(best_format)
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor="red", lw=2))

        plt.title("Format Comparison by Quality Range", fontsize=14)
        plt.ylabel("Quality Range (SSIMULACRA2 Score)", fontsize=12)

    else:
        # Fallback without seaborn
        ax = pivot_data.plot(kind="bar", figsize=(10, 6))
        plt.title("Format Comparison by Quality Range")
        plt.ylabel("Compression Ratio (higher is better)")
        plt.grid(axis="y", alpha=0.3)
        plt.legend(title="Format")

        # Format y ticks
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}x"))

    # Save or show
    if output_path:
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved format comparison plot to {output_path}")
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

    return True


def plot_implementation_comparison(df, output_path=None):
    """Create scatter plots comparing different SSIMULACRA2 implementations.

    Args:
        df: DataFrame with benchmark data
        output_path: Path to save the plot image

    Returns:
        bool: True if successful, False otherwise
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib is not available. Cannot create plot.")
        return False

    # Check which implementations have data
    implementations = ["python", "cpp", "rust"]
    available_impls = [
        impl
        for impl in implementations
        if f"{impl}_score" in df.columns and df[f"{impl}_score"].notna().sum() > 5
    ]

    if len(available_impls) < 2:
        print("Not enough implementation data to compare")
        return False

    # Create grid for comparison plots
    num_comparisons = len(available_impls) * (len(available_impls) - 1) // 2
    fig, axes = plt.subplots(
        1, num_comparisons, figsize=(6 * num_comparisons, 6), squeeze=False
    )
    axes = axes.flatten()

    # Use seaborn for prettier plots if available
    if SEABORN_AVAILABLE:
        sns.set_style("whitegrid")

    # Generate all pairwise comparisons
    plot_idx = 0
    for i, impl1 in enumerate(available_impls):
        for impl2 in available_impls[i + 1 :]:
            if plot_idx >= len(axes):
                break

            # Get column names
            col1 = f"{impl1}_score"
            col2 = f"{impl2}_score"

            # Get data points where both implementations have results
            mask = df[col1].notna() & df[col2].notna()
            x = df.loc[mask, col1]
            y = df.loc[mask, col2]

            if len(x) < 5:
                axes[plot_idx].text(
                    0.5, 0.5, "Insufficient data", ha="center", va="center", fontsize=12
                )
                plot_idx += 1
                continue

            ax = axes[plot_idx]

            # Plot scatter with regression line
            if SEABORN_AVAILABLE:
                sns.regplot(
                    x=x,
                    y=y,
                    ax=ax,
                    scatter_kws={"alpha": 0.5},
                    line_kws={"color": "red"},
                )
            else:
                ax.scatter(x, y, alpha=0.5)

                # Add regression line
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                x_sorted = np.sort(x)
                ax.plot(x_sorted, p(x_sorted), "r-")

            # Add identity line (x=y)
            lims = [
                min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1]),
            ]
            ax.plot(lims, lims, "k--", alpha=0.5, zorder=0)
            ax.set_xlim(lims)
            ax.set_ylim(lims)

            # Add correlation coefficient
            corr = x.corr(y)
            ax.annotate(
                f"r = {corr:.3f}",
                xy=(0.05, 0.95),
                xycoords="axes fraction",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
            )

            # Add mean difference
            mean_diff = (y - x).mean()
            ax.annotate(
                f"Mean diff: {mean_diff:.3f}",
                xy=(0.05, 0.85),
                xycoords="axes fraction",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
            )

            # Set labels
            ax.set_xlabel(f"{impl1} Score")
            ax.set_ylabel(f"{impl2} Score")
            ax.set_title(f"{impl1} vs {impl2}")

            plot_idx += 1

    # Set figure title
    plt.suptitle("SSIMULACRA2 Implementation Comparison", fontsize=16)

    # Save or show
    if output_path:
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved implementation comparison plot to {output_path}")
        plt.close()
    else:
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
        plt.show()

    return True


def plot_quality_distribution(df, output_path=None):
    """Create plots showing quality score distribution by format.

    Args:
        df: DataFrame with benchmark data
        output_path: Path to save the plot image

    Returns:
        bool: True if successful, False otherwise
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib is not available. Cannot create plot.")
        return False

    # Get score column
    try:
        score_col = get_best_score_column(df)
    except ValueError:
        print("No valid score column available")
        return False

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    if SEABORN_AVAILABLE:
        # Use seaborn for prettier plots
        sns.set_style("whitegrid")

        # Create violinplot
        ax = sns.violinplot(
            x="compression_type",
            y=score_col,
            data=df,
            palette="viridis",
            inner="quartile",
        )

        # Add individual points (if not too many)
        if len(df) < 500:
            sns.stripplot(
                x="compression_type",
                y=score_col,
                data=df,
                color="black",
                alpha=0.3,
                size=3,
                jitter=True,
            )

    else:
        # Fallback without seaborn
        formats = df["compression_type"].unique()
        for i, fmt in enumerate(formats):
            subset = df[df["compression_type"] == fmt]
            if len(subset) > 0:
                # Create boxplot
                ax.boxplot(subset[score_col].dropna(), positions=[i], widths=0.5)

        ax.set_xticks(range(len(formats)))
        ax.set_xticklabels(formats)

    # Add target quality lines
    for quality in [60, 70, 80, 90]:
        ax.axhline(y=quality, color="red", linestyle="--", alpha=0.3)
        ax.text(
            ax.get_xlim()[1],
            quality,
            f"{quality}",
            ha="right",
            va="center",
            color="red",
            alpha=0.7,
        )

    # Set labels and title
    ax.set_title("Quality Score Distribution by Format", fontsize=14)
    ax.set_xlabel("Format", fontsize=12)
    ax.set_ylabel("SSIMULACRA2 Score", fontsize=12)

    # Add grid
    ax.grid(axis="y", alpha=0.3)

    # Save or show
    if output_path:
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved quality distribution plot to {output_path}")
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

    return True


def create_dashboard(df, output_dir):
    """Create a complete set of visualization plots.

    Args:
        df: DataFrame with benchmark data
        output_dir: Directory to save plots

    Returns:
        list: Paths to generated plots
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib is not available. Cannot create dashboard.")
        return []

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate all plots
    plots = []

    # Quality vs Compression
    quality_plot = os.path.join(output_dir, "quality_vs_compression.png")
    if plot_quality_vs_compression(df, quality_plot):
        plots.append(quality_plot)

    # Format Comparison
    format_plot = os.path.join(output_dir, "format_comparison.png")
    if plot_format_comparison(df, format_plot):
        plots.append(format_plot)

    # Implementation Comparison
    impl_plot = os.path.join(output_dir, "implementation_comparison.png")
    if plot_implementation_comparison(df, impl_plot):
        plots.append(impl_plot)

    # Quality Distribution
    dist_plot = os.path.join(output_dir, "quality_distribution.png")
    if plot_quality_distribution(df, dist_plot):
        plots.append(dist_plot)

    return plots
