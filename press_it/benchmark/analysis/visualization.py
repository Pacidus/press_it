"""Visualization utilities for benchmark analysis."""

from logging import warn
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from scipy import stats
import warnings

# Set up matplotlib for high-quality figures with dark theme
plt.style.use("dark_background")

# Configure matplotlib for dark theme (without LaTeX initially)
mpl.rcParams.update(
    {
        "figure.figsize": (12, 8),
        "savefig.dpi": 300,
        "font.size": 12,
        "text.color": "white",
        "axes.labelcolor": "white",
        "axes.edgecolor": "#999999",
        "axes.facecolor": "#1e1e1e",
        "axes.grid": True,
        "grid.color": "#333333",
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
        "figure.facecolor": "#121212",
        "savefig.facecolor": "#121212",
        "lines.linewidth": 2,
        "xtick.color": "white",
        "ytick.color": "white",
        "legend.facecolor": "#2a2a2a",
        "legend.edgecolor": "#999999",
        "legend.framealpha": 0.8,
        "legend.fontsize": 10,
        "legend.title_fontsize": 11,
        "patch.edgecolor": "#2a2a2a",
    }
)

# Check if LaTeX is available but don't use it by default
USE_LATEX = False
try:
    # Test if LaTeX is working
    test_fig = plt.figure()
    test_ax = test_fig.add_subplot(111)
    test_ax.text(0.5, 0.5, r"$\alpha$", usetex=True)
    plt.close(test_fig)
    USE_LATEX = True  # LaTeX is working
    mpl.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
        }
    )

    print("LaTeX is available for fancy math formatting")
except Exception:
    print("LaTeX not available or working, using standard fonts")

# For HTML visualizations
try:
    import bokeh

    from bokeh.plotting import figure, output_file, save
    from bokeh.layouts import column, row, gridplot
    from bokeh.models import ColumnDataSource, HoverTool, Legend, LegendItem
    from bokeh.palettes import Category10, Viridis256
    from bokeh.themes import built_in_themes
    from bokeh.transform import dodge

    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False
    warnings.warn("Bokeh not available. HTML visualizations will be disabled.")

# Define a dark theme for Bokeh
DARK_THEME = {
    "attrs": {
        "Figure": {
            "background_fill_color": "#121212",
            "border_fill_color": "#121212",
            "outline_line_color": "#444444",
        },
        "Axis": {
            "axis_line_color": "#666666",
            "major_tick_line_color": "#666666",
            "minor_tick_line_color": "#444444",
            "major_label_text_color": "#DDDDDD",
        },
        "Grid": {
            "grid_line_color": "#444444",
            "grid_line_alpha": 0.7,
        },
        "Title": {
            "text_color": "#FFFFFF",
        },
        "Legend": {
            "background_fill_color": "#2a2a2a",
            "border_line_color": "#666666",
            "border_line_alpha": 0.8,
            "text_color": "#DDDDDD",
            "title_text_color": "#FFFFFF",
        },
    }
}

# Define better color palettes for dark theme
DARK_COLORS = [
    "#FF9500",  # Orange
    "#00BFFF",  # Bright Blue
    "#FF3B30",  # Bright Red
    "#4CD964",  # Bright Green
    "#5856D6",  # Purple
    "#FFCC00",  # Yellow
    "#34AADC",  # Light Blue
    "#FF2D55",  # Pink
    "#FFFFFF",  # White
    "#5AC8FA",  # Aqua
]


def ensure_output_dir(output_dir):
    """Ensure the output directory exists."""
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def prepare_data(df):
    """Prepare and clean the benchmark data for visualization.

    Args:
        df: Benchmark dataframe

    Returns:
        pd.DataFrame: Cleaned dataframe with additional columns
    """
    # Add compression_ratio if not present
    if (
        "compression_ratio" not in df.columns
        and "original_size" in df.columns
        and "compressed_size" in df.columns
    ):
        df["compression_ratio"] = df["original_size"] / df["compressed_size"]

    # Make sure compression_type is a category
    if "compression_type" in df.columns:
        df["compression_type"] = df["compression_type"].astype("category")

    # Remove extreme outliers in compression_ratio (e.g., if compressed_size is near zero)
    if "compression_ratio" in df.columns:
        # Get reasonable limits for compression ratio (using quantiles)
        q1, q3 = df["compression_ratio"].quantile([0.01, 0.99])
        iqr = q3 - q1
        upper_limit = q3 + 3 * iqr

        # Filter out extreme values
        df = df[df["compression_ratio"] <= upper_limit]

    return df


def format_poly_formula(coeffs, decimal_places=4):
    """Format polynomial coefficients into a readable formula without LaTeX.

    Args:
        coeffs: Array of polynomial coefficients (highest power first)
        decimal_places: Number of decimal places to show

    Returns:
        str: Formatted polynomial formula
    """
    degree = len(coeffs) - 1
    terms = []

    for i, coef in enumerate(coeffs):
        power = degree - i
        coef_str = f"{coef:.{decimal_places}f}"

        # Skip zero coefficients except for the constant term
        if coef == 0 and power > 0:
            continue

        # Format the term based on its power
        if power == 0:
            # Constant term
            terms.append(coef_str)
        elif power == 1:
            # Linear term (x)
            if coef == 1:
                terms.append("x")
            elif coef == -1:
                terms.append("-x")
            else:
                terms.append(f"{coef_str}x")
        else:
            # Higher power terms (x^2, x^3, etc.)
            if coef == 1:
                terms.append(f"x^{power}")
            elif coef == -1:
                terms.append(f"-x^{power}")
            else:
                terms.append(f"{coef_str}x^{power}")

    # Join terms with appropriate signs
    formula = ""
    for i, term in enumerate(terms):
        if i == 0:
            # First term
            formula += term
        else:
            # Add sign as prefix to subsequent terms
            if term[0] == "-":
                formula += f" - {term[1:]}"
            else:
                formula += f" + {term}"

    return "$" + formula + "$"


def fit_regression_model(x, y, model_type="linear", degree=2):
    """Fit a regression model to the data.

    Args:
        x: Independent variable
        y: Dependent variable
        model_type: Type of regression model ('linear', 'polynomial', 'log')
        degree: Polynomial degree if model_type is 'polynomial'

    Returns:
        dict: Model information including function, r2, and parameters
    """
    # Remove NaN values
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_clean = x[mask]
    y_clean = y[mask]

    if len(x_clean) < 3:
        return None  # Not enough data points

    result = {"x": x_clean, "y": y_clean, "type": model_type}

    if model_type == "linear":
        # Linear regression: y = ax + b
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
        result["func"] = lambda x_new: slope * x_new + intercept

        # Format the formula
        if slope >= 0:
            result["formula"] = f"y = {slope:.4f}x + {intercept:.4f}"
        else:
            result["formula"] = f"y = {slope:.4f}x - {abs(intercept):.4f}"

        result["r2"] = r_value**2
        result["params"] = {"slope": slope, "intercept": intercept}

    elif model_type == "polynomial":
        # Polynomial regression: y = a_n*x^n + ... + a_1*x + a_0
        coeffs = np.polyfit(x_clean, y_clean, degree)
        poly = np.poly1d(coeffs)

        # Calculate R^2
        y_pred = poly(x_clean)
        ss_total = np.sum((y_clean - np.mean(y_clean)) ** 2)
        ss_residual = np.sum((y_clean - y_pred) ** 2)
        r2 = 1 - (ss_residual / ss_total)

        result["func"] = poly

        # Format formula without LaTeX for safety
        formula = "y = " + format_poly_formula(coeffs)
        result["formula"] = formula

        result["r2"] = r2
        result["params"] = {"coefficients": coeffs}
        result["degree"] = degree

    elif model_type == "log":
        # Logarithmic regression: y = a*log(x) + b
        # Filter out non-positive x values
        positive_mask = x_clean > 0
        if np.sum(positive_mask) < 3:
            return None  # Not enough data points after filtering

        x_log = x_clean[positive_mask]
        y_log = y_clean[positive_mask]

        slope, intercept, r_value, p_value, std_err = stats.linregress(
            np.log(x_log), y_log
        )
        result["func"] = lambda x_new: slope * np.log(x_new) + intercept

        # Format the formula
        if intercept >= 0:
            result["formula"] = f"y = {slope:.4f}ln(x) + {intercept:.4f}"
        else:
            result["formula"] = f"y = {slope:.4f}ln(x) - {abs(intercept):.4f}"

        result["r2"] = r_value**2
        result["params"] = {"slope": slope, "intercept": intercept}

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return result


def format_comparison_plots(df, output_dir, formats=None):
    """Generate plots comparing different compression formats.

    Args:
        df: Benchmark dataframe
        output_dir: Directory to save plots
        formats: List of output formats ('png', 'html', or both)

    Returns:
        list: Paths to generated figures
    """
    # Default to both formats if not specified
    if formats is None:
        formats = ["png", "html"]

    # Prepare data
    data = prepare_data(df)

    # Ensure 'cpp_score' is available (reference implementation)
    if "cpp_score" not in data.columns or data["cpp_score"].isna().all():
        print("Warning: No C++ SSIMULACRA2 scores available. Using available scores.")
        for col in ["python_score", "rust_score"]:
            if col in data.columns and not data[col].isna().all():
                data["reference_score"] = data[col]
                print(f"Using {col} as reference score.")
                break
        else:
            raise ValueError("No valid SSIMULACRA2 scores available in the dataset.")
    else:
        data["reference_score"] = data["cpp_score"]

    # List to store figure paths
    figure_paths = []

    # Make sure output directory exists
    ensure_output_dir(output_dir)

    # Create figures
    figure_paths.extend(_plot_quality_vs_parameter(data, output_dir, formats))
    figure_paths.extend(_plot_ratio_vs_parameter(data, output_dir, formats))
    figure_paths.extend(_plot_quality_vs_ratio(data, output_dir, formats))

    return figure_paths


def implementation_comparison_plots(df, output_dir, formats=None):
    """Generate plots comparing different SSIMULACRA2 implementations.

    Args:
        df: Benchmark dataframe
        output_dir: Directory to save plots
        formats: List of output formats ('png', 'html', or both)

    Returns:
        list: Paths to generated figures
    """
    # Default to both formats if not specified
    if formats is None:
        formats = ["png", "html"]

    # Prepare data
    data = prepare_data(df)

    # List to store figure paths
    figure_paths = []

    # Make sure output directory exists
    ensure_output_dir(output_dir)

    # Compare implementations
    if "cpp_score" in data.columns and "python_score" in data.columns:
        # Check if we have sufficient non-null data
        valid_data = data.dropna(subset=["cpp_score", "python_score"])
        if len(valid_data) > 5:  # Arbitrary threshold for "enough data"
            figure_paths.extend(
                _plot_implementation_correlation(
                    data,
                    "cpp_score",
                    "python_score",
                    "C++",
                    "Python",
                    output_dir,
                    formats,
                )
            )

    if "cpp_score" in data.columns and "rust_score" in data.columns:
        valid_data = data.dropna(subset=["cpp_score", "rust_score"])
        if len(valid_data) > 5:
            figure_paths.extend(
                _plot_implementation_correlation(
                    data, "cpp_score", "rust_score", "C++", "Rust", output_dir, formats
                )
            )

    if "python_score" in data.columns and "rust_score" in data.columns:
        valid_data = data.dropna(subset=["python_score", "rust_score"])
        if len(valid_data) > 5:
            figure_paths.extend(
                _plot_implementation_correlation(
                    data,
                    "python_score",
                    "rust_score",
                    "Python",
                    "Rust",
                    output_dir,
                    formats,
                )
            )

    # Plot distribution of differences
    figure_paths.extend(_plot_implementation_differences(data, output_dir, formats))

    # Plot implementation performance by format
    figure_paths.extend(_plot_implementation_by_format(data, output_dir, formats))

    return figure_paths


def _plot_quality_vs_parameter(data, output_dir, formats):
    """Plot quality scores vs compression parameter for each format."""
    figure_paths = []

    # Check if we have the necessary columns
    if not all(
        col in data.columns
        for col in ["compression_type", "quality", "reference_score"]
    ):
        print("Warning: Missing required columns for quality vs parameter plot")
        return figure_paths

    # Filter out rows with missing values
    plot_data = data.dropna(subset=["compression_type", "quality", "reference_score"])

    # Make sure we have data to plot
    if len(plot_data) == 0:
        print("Warning: No valid data for quality vs parameter plot")
        return figure_paths

    # Get unique compression types
    compression_types = plot_data["compression_type"].unique()

    # Create the matplotlib figure
    if "png" in formats:
        plt.figure(figsize=(12, 8))

        # Plot each compression type with upgraded colors for dark theme
        for i, comp_type in enumerate(compression_types):
            type_data = plot_data[plot_data["compression_type"] == comp_type]
            plt.scatter(
                type_data["quality"],
                type_data["reference_score"],
                color=DARK_COLORS[i % len(DARK_COLORS)],
                alpha=0.7,
                s=50,  # Larger point size
                label=comp_type,
                edgecolor="#121212",  # Dark edge to improve visibility
                linewidth=0.5,
            )

            # Fit regression model
            model = fit_regression_model(
                type_data["quality"].values,
                type_data["reference_score"].values,
                model_type="polynomial",
                degree=3,
            )

            if model:
                # Add regression line
                x_range = np.linspace(
                    type_data["quality"].min(), type_data["quality"].max(), 100
                )
                plt.plot(
                    x_range,
                    model["func"](x_range),
                    color=DARK_COLORS[i % len(DARK_COLORS)],
                    linestyle="--",
                    linewidth=2,
                    alpha=0.9,
                )

                # Add formula to plot (safely formatted)
                plt.text(
                    0.02,
                    0.96 - 0.05 * i,
                    f"{comp_type}: {model['formula']} (R² = {model['r2']:.3f})",
                    transform=plt.gca().transAxes,
                    fontsize=10,
                    color=DARK_COLORS[i % len(DARK_COLORS)],
                    bbox=dict(
                        facecolor="#2a2a2a",
                        alpha=0.7,
                        edgecolor="#444444",
                        boxstyle="round,pad=0.5",
                    ),
                )

        plt.xlabel("Compression Quality Parameter")
        plt.ylabel("SSIMULACRA2 Score (C++ Implementation)")
        plt.title("Perceptual Quality vs. Compression Parameter")
        plt.legend(title="Format", loc="lower right", framealpha=0.7)
        plt.grid(True, alpha=0.4, linestyle="--", color="#444444")
        plt.tight_layout()

        # Save figure
        fig_path = os.path.join(output_dir, "quality_vs_parameter.png")
        plt.savefig(fig_path, bbox_inches="tight")
        plt.close()
        figure_paths.append(fig_path)

    # Create the Bokeh figure
    if "html" in formats and BOKEH_AVAILABLE:
        output_file(os.path.join(output_dir, "quality_vs_parameter.html"))

        p = figure(
            title="Perceptual Quality vs. Compression Parameter",
            x_axis_label="Compression Quality Parameter",
            y_axis_label="SSIMULACRA2 Score",
            tooltips=[
                ("Format", "@format"),
                ("Quality", "@quality"),
                ("SSIMULACRA2", "@score"),
                ("Image Size", "@size"),
            ],
            width=900,
            height=500,
            background_fill_color="#121212",
            border_fill_color="#121212",
            outline_line_color="#444444",
        )

        # Add dark theme styling
        p.title.text_color = "white"
        p.xaxis.axis_label_text_color = "white"
        p.yaxis.axis_label_text_color = "white"
        p.xaxis.major_label_text_color = "white"
        p.yaxis.major_label_text_color = "white"
        p.xgrid.grid_line_color = "#333333"
        p.ygrid.grid_line_color = "#333333"

        # Plot each compression type
        for i, comp_type in enumerate(compression_types):
            type_data = plot_data[plot_data["compression_type"] == comp_type]

            source = ColumnDataSource(
                data=dict(
                    quality=type_data["quality"],
                    score=type_data["reference_score"],
                    format=[comp_type] * len(type_data),
                    size=(
                        type_data["width"] * type_data["height"]
                        if all(col in type_data.columns for col in ["width", "height"])
                        else [0] * len(type_data)
                    ),
                )
            )

            # Add points - use scatter instead of circle
            p.scatter(
                "quality",
                "score",
                source=source,
                size=10,
                color=DARK_COLORS[i % len(DARK_COLORS)],
                alpha=0.7,
                legend_label=comp_type,
            )

            # Fit regression model
            model = fit_regression_model(
                type_data["quality"].values,
                type_data["reference_score"].values,
                model_type="polynomial",
                degree=3,
            )

            if model:
                # Add regression line
                x_range = np.linspace(
                    type_data["quality"].min(), type_data["quality"].max(), 100
                )
                p.line(
                    x_range,
                    model["func"](x_range),
                    color=DARK_COLORS[i % len(DARK_COLORS)],
                    line_dash="dashed",
                    line_width=2.5,
                    alpha=0.9,
                    legend_label=f"{comp_type} (fit)",
                )

        p.legend.title = "Format"
        p.legend.location = "top_left"
        p.legend.background_fill_color = "#2a2a2a"
        p.legend.background_fill_alpha = 0.7
        p.legend.border_line_color = "#666666"
        p.legend.label_text_color = "white"
        p.legend.title_text_color = "white"

        # Save the HTML figure
        save(p)
        figure_paths.append(os.path.join(output_dir, "quality_vs_parameter.html"))

    return figure_paths


def _plot_ratio_vs_parameter(data, output_dir, formats):
    """Plot compression ratio vs compression parameter for each format."""
    figure_paths = []

    # Check if we have the necessary columns
    if not all(
        col in data.columns
        for col in ["compression_type", "quality", "compression_ratio"]
    ):
        print("Warning: Missing required columns for ratio vs parameter plot")
        return figure_paths

    # Filter out rows with missing values
    plot_data = data.dropna(subset=["compression_type", "quality", "compression_ratio"])

    # Make sure we have data to plot
    if len(plot_data) == 0:
        print("Warning: No valid data for ratio vs parameter plot")
        return figure_paths

    # Get unique compression types
    compression_types = plot_data["compression_type"].unique()

    # Create the matplotlib figure
    if "png" in formats:
        plt.figure(figsize=(12, 8))

        # Plot each compression type
        for i, comp_type in enumerate(compression_types):
            type_data = plot_data[plot_data["compression_type"] == comp_type]
            plt.scatter(
                type_data["quality"],
                type_data["compression_ratio"],
                color=DARK_COLORS[i % len(DARK_COLORS)],
                alpha=0.7,
                s=50,
                label=comp_type,
                edgecolor="#121212",
                linewidth=0.5,
            )

            # Fit regression model
            model = fit_regression_model(
                type_data["quality"].values,
                type_data["compression_ratio"].values,
                model_type="polynomial",
                degree=2,
            )

            if model:
                # Add regression line
                x_range = np.linspace(
                    type_data["quality"].min(), type_data["quality"].max(), 100
                )
                plt.plot(
                    x_range,
                    model["func"](x_range),
                    color=DARK_COLORS[i % len(DARK_COLORS)],
                    linestyle="--",
                    linewidth=2,
                    alpha=0.9,
                )

                # Add formula to plot
                plt.text(
                    0.02,
                    0.96 - 0.05 * i,
                    f"{comp_type}: {model['formula']} (R² = {model['r2']:.3f})",
                    transform=plt.gca().transAxes,
                    fontsize=10,
                    color=DARK_COLORS[i % len(DARK_COLORS)],
                    bbox=dict(
                        facecolor="#2a2a2a",
                        alpha=0.7,
                        edgecolor="#444444",
                        boxstyle="round,pad=0.5",
                    ),
                )

        plt.xlabel("Compression Quality Parameter")
        plt.ylabel("Compression Ratio (Original / Compressed)")
        plt.title("Compression Ratio vs. Quality Parameter")
        plt.legend(title="Format", loc="upper left", framealpha=0.7)
        plt.grid(True, alpha=0.4, linestyle="--", color="#444444")
        plt.tight_layout()

        # Save figure
        fig_path = os.path.join(output_dir, "ratio_vs_parameter.png")
        plt.savefig(fig_path, bbox_inches="tight")
        plt.close()
        figure_paths.append(fig_path)

    # Create the Bokeh figure
    if "html" in formats and BOKEH_AVAILABLE:
        output_file(os.path.join(output_dir, "ratio_vs_parameter.html"))

        p = figure(
            title="Compression Ratio vs. Quality Parameter",
            x_axis_label="Compression Quality Parameter",
            y_axis_label="Compression Ratio (Original / Compressed)",
            tooltips=[
                ("Format", "@format"),
                ("Quality", "@quality"),
                ("Ratio", "@ratio{0.00}x"),
                ("Original Size", "@original{0,0} bytes"),
                ("Compressed Size", "@compressed{0,0} bytes"),
            ],
            width=900,
            height=500,
            background_fill_color="#121212",
            border_fill_color="#121212",
            outline_line_color="#444444",
        )

        # Add dark theme styling
        p.title.text_color = "white"
        p.xaxis.axis_label_text_color = "white"
        p.yaxis.axis_label_text_color = "white"
        p.xaxis.major_label_text_color = "white"
        p.yaxis.major_label_text_color = "white"
        p.xgrid.grid_line_color = "#333333"
        p.ygrid.grid_line_color = "#333333"

        # Plot each compression type
        for i, comp_type in enumerate(compression_types):
            type_data = plot_data[plot_data["compression_type"] == comp_type]

            source = ColumnDataSource(
                data=dict(
                    quality=type_data["quality"],
                    ratio=type_data["compression_ratio"],
                    format=[comp_type] * len(type_data),
                    original=(
                        type_data["original_size"]
                        if "original_size" in type_data.columns
                        else [0] * len(type_data)
                    ),
                    compressed=(
                        type_data["compressed_size"]
                        if "compressed_size" in type_data.columns
                        else [0] * len(type_data)
                    ),
                )
            )

            # Add points - use scatter instead of circle
            p.scatter(
                "quality",
                "ratio",
                source=source,
                size=10,
                color=DARK_COLORS[i % len(DARK_COLORS)],
                alpha=0.7,
                legend_label=comp_type,
            )

            # Fit regression model
            model = fit_regression_model(
                type_data["quality"].values,
                type_data["compression_ratio"].values,
                model_type="polynomial",
                degree=2,
            )

            if model:
                # Add regression line
                x_range = np.linspace(
                    type_data["quality"].min(), type_data["quality"].max(), 100
                )
                p.line(
                    x_range,
                    model["func"](x_range),
                    color=DARK_COLORS[i % len(DARK_COLORS)],
                    line_dash="dashed",
                    line_width=2.5,
                    alpha=0.9,
                    legend_label=f"{comp_type} (fit)",
                )

        p.legend.title = "Format"
        p.legend.location = "top_left"
        p.legend.background_fill_color = "#2a2a2a"
        p.legend.background_fill_alpha = 0.7
        p.legend.border_line_color = "#666666"
        p.legend.label_text_color = "white"
        p.legend.title_text_color = "white"

        # Save the HTML figure
        save(p)
        figure_paths.append(os.path.join(output_dir, "ratio_vs_parameter.html"))

    return figure_paths


def _plot_quality_vs_ratio(data, output_dir, formats):
    """Plot quality scores vs compression ratio for each format."""
    figure_paths = []

    # Check if we have the necessary columns
    if not all(
        col in data.columns
        for col in ["compression_type", "compression_ratio", "reference_score"]
    ):
        print("Warning: Missing required columns for quality vs ratio plot")
        return figure_paths

    # Filter out rows with missing values
    plot_data = data.dropna(
        subset=["compression_type", "compression_ratio", "reference_score"]
    )

    # Make sure we have data to plot
    if len(plot_data) == 0:
        print("Warning: No valid data for quality vs ratio plot")
        return figure_paths

    # Get unique compression types
    compression_types = plot_data["compression_type"].unique()

    # Create the matplotlib figure
    if "png" in formats:
        plt.figure(figsize=(12, 8))

        # Plot each compression type
        for i, comp_type in enumerate(compression_types):
            type_data = plot_data[plot_data["compression_type"] == comp_type]
            plt.scatter(
                type_data["compression_ratio"],
                type_data["reference_score"],
                color=DARK_COLORS[i % len(DARK_COLORS)],
                alpha=0.7,
                s=50,
                label=comp_type,
                edgecolor="#121212",
                linewidth=0.5,
            )

            # Fit regression model
            model = fit_regression_model(
                type_data["compression_ratio"].values,
                type_data["reference_score"].values,
                model_type="log",
            )

            if model:
                # Add regression line
                x_range = np.linspace(
                    type_data["compression_ratio"].min(),
                    type_data["compression_ratio"].max(),
                    100,
                )
                plt.plot(
                    x_range,
                    model["func"](x_range),
                    color=DARK_COLORS[i % len(DARK_COLORS)],
                    linestyle="--",
                    linewidth=2,
                    alpha=0.9,
                )

                # Add formula to plot
                plt.text(
                    0.02,
                    0.96 - 0.05 * i,
                    f"{comp_type}: {model['formula']} (R² = {model['r2']:.3f})",
                    transform=plt.gca().transAxes,
                    fontsize=10,
                    color=DARK_COLORS[i % len(DARK_COLORS)],
                    bbox=dict(
                        facecolor="#2a2a2a",
                        alpha=0.7,
                        edgecolor="#444444",
                        boxstyle="round,pad=0.5",
                    ),
                )

        plt.xlabel("Compression Ratio (Original / Compressed)")
        plt.ylabel("SSIMULACRA2 Score (C++ Implementation)")
        plt.title("Perceptual Quality vs. Compression Ratio")
        plt.legend(title="Format", loc="upper right", framealpha=0.7)
        plt.grid(True, alpha=0.4, linestyle="--", color="#444444")
        plt.tight_layout()

        # Save figure
        fig_path = os.path.join(output_dir, "quality_vs_ratio.png")
        plt.savefig(fig_path, bbox_inches="tight")
        plt.close()
        figure_paths.append(fig_path)

    # Create the Bokeh figure
    if "html" in formats and BOKEH_AVAILABLE:
        output_file(os.path.join(output_dir, "quality_vs_ratio.html"))

        p = figure(
            title="Perceptual Quality vs. Compression Ratio",
            x_axis_label="Compression Ratio (Original / Compressed)",
            y_axis_label="SSIMULACRA2 Score",
            tooltips=[
                ("Format", "@format"),
                ("Ratio", "@ratio{0.00}x"),
                ("SSIMULACRA2", "@score"),
                ("Quality", "@quality"),
            ],
            width=900,
            height=500,
            background_fill_color="#121212",
            border_fill_color="#121212",
            outline_line_color="#444444",
        )

        # Add dark theme styling
        p.title.text_color = "white"
        p.xaxis.axis_label_text_color = "white"
        p.yaxis.axis_label_text_color = "white"
        p.xaxis.major_label_text_color = "white"
        p.yaxis.major_label_text_color = "white"
        p.xgrid.grid_line_color = "#333333"
        p.ygrid.grid_line_color = "#333333"

        # Plot each compression type
        for i, comp_type in enumerate(compression_types):
            type_data = plot_data[plot_data["compression_type"] == comp_type]

            source = ColumnDataSource(
                data=dict(
                    ratio=type_data["compression_ratio"],
                    score=type_data["reference_score"],
                    format=[comp_type] * len(type_data),
                    quality=(
                        type_data["quality"]
                        if "quality" in type_data.columns
                        else [0] * len(type_data)
                    ),
                )
            )

            # Add points - use scatter instead of circle
            p.scatter(
                "ratio",
                "score",
                source=source,
                size=10,
                color=DARK_COLORS[i % len(DARK_COLORS)],
                alpha=0.7,
                legend_label=comp_type,
            )

            # Fit regression model
            model = fit_regression_model(
                type_data["compression_ratio"].values,
                type_data["reference_score"].values,
                model_type="log",
            )

            if model:
                # Add regression line
                x_range = np.linspace(
                    type_data["compression_ratio"].min(),
                    type_data["compression_ratio"].max(),
                    100,
                )
                p.line(
                    x_range,
                    model["func"](x_range),
                    color=DARK_COLORS[i % len(DARK_COLORS)],
                    line_dash="dashed",
                    line_width=2.5,
                    alpha=0.9,
                    legend_label=f"{comp_type} (fit)",
                )

        p.legend.title = "Format"
        p.legend.location = "bottom_right"
        p.legend.background_fill_color = "#2a2a2a"
        p.legend.background_fill_alpha = 0.7
        p.legend.border_line_color = "#666666"
        p.legend.label_text_color = "white"
        p.legend.title_text_color = "white"

        # Save the HTML figure
        save(p)
        figure_paths.append(os.path.join(output_dir, "quality_vs_ratio.html"))

    return figure_paths


def _plot_implementation_correlation(
    data, impl1_col, impl2_col, impl1_name, impl2_name, output_dir, formats
):
    """Plot correlation between two SSIMULACRA2 implementations."""
    figure_paths = []

    # Filter out rows with missing values
    plot_data = data.dropna(subset=[impl1_col, impl2_col])

    # Make sure we have data to plot
    if len(plot_data) == 0:
        print(
            f"Warning: No valid data for {impl1_name} vs {impl2_name} correlation plot"
        )
        return figure_paths

    # Create the matplotlib figure
    if "png" in formats:
        plt.figure(figsize=(10, 10))

        # Scatter plot with colors by compression type
        if "compression_type" in plot_data.columns:
            # Get unique compression types for coloring
            comp_types = plot_data["compression_type"].unique()

            for i, comp_type in enumerate(comp_types):
                type_data = plot_data[plot_data["compression_type"] == comp_type]
                plt.scatter(
                    type_data[impl1_col],
                    type_data[impl2_col],
                    color=DARK_COLORS[i % len(DARK_COLORS)],
                    alpha=0.7,
                    s=50,
                    label=comp_type,
                    edgecolor="#121212",
                    linewidth=0.5,
                )
        else:
            # Single color if no format information
            plt.scatter(
                plot_data[impl1_col],
                plot_data[impl2_col],
                color=DARK_COLORS[0],
                alpha=0.7,
                s=50,
                edgecolor="#121212",
                linewidth=0.5,
            )

        # Fit linear regression model
        model = fit_regression_model(
            plot_data[impl1_col].values,
            plot_data[impl2_col].values,
            model_type="linear",
        )

        if model:
            # Add regression line
            x_range = np.linspace(
                plot_data[impl1_col].min(), plot_data[impl1_col].max(), 100
            )
            plt.plot(
                x_range,
                model["func"](x_range),
                "r-",
                linewidth=2.5,
                label=f"Regression: {model['formula']}\nR² = {model['r2']:.3f}",
            )

            # Add perfect correlation line (y = x)
            plt.plot(
                [0, 100],
                [0, 100],
                color="white",
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
                label="Perfect correlation (y = x)",
            )

        plt.xlabel(f"{impl1_name} SSIMULACRA2 Score")
        plt.ylabel(f"{impl2_name} SSIMULACRA2 Score")
        plt.title(
            f"Correlation between {impl1_name} and {impl2_name} SSIMULACRA2 Implementations"
        )
        plt.legend(framealpha=0.8)
        plt.grid(True, alpha=0.4, linestyle="--", color="#444444")
        plt.axis("equal")
        plt.tight_layout()

        # Save figure
        fig_path = os.path.join(
            output_dir, f"{impl1_name.lower()}_{impl2_name.lower()}_correlation.png"
        )
        plt.savefig(fig_path, bbox_inches="tight")
        plt.close()
        figure_paths.append(fig_path)

    # Create the Bokeh figure
    if "html" in formats and BOKEH_AVAILABLE:
        output_file(
            os.path.join(
                output_dir,
                f"{impl1_name.lower()}_{impl2_name.lower()}_correlation.html",
            )
        )

        p = figure(
            title=f"Correlation between {impl1_name} and {impl2_name} SSIMULACRA2 Implementations",
            x_axis_label=f"{impl1_name} SSIMULACRA2 Score",
            y_axis_label=f"{impl2_name} SSIMULACRA2 Score",
            tooltips=[
                ("Format", "@format"),
                (f"{impl1_name}", f"@{impl1_col}"),
                (f"{impl2_name}", f"@{impl2_col}"),
                ("Difference", "@diff"),
                ("Quality", "@quality"),
            ],
            width=800,
            height=800,
            background_fill_color="#121212",
            border_fill_color="#121212",
            outline_line_color="#444444",
        )

        # Add dark theme styling
        p.title.text_color = "white"
        p.xaxis.axis_label_text_color = "white"
        p.yaxis.axis_label_text_color = "white"
        p.xaxis.major_label_text_color = "white"
        p.yaxis.major_label_text_color = "white"
        p.xgrid.grid_line_color = "#333333"
        p.ygrid.grid_line_color = "#333333"

        # Add points
        if "compression_type" in plot_data.columns:
            # Color by format
            compression_types = plot_data["compression_type"].unique()

            for i, comp_type in enumerate(compression_types):
                # Safely filter the data - use boolean indexing
                type_mask = plot_data["compression_type"] == comp_type
                if type_mask.any():  # Check if any rows match this format
                    type_data = plot_data[type_mask]

                    type_source = ColumnDataSource(
                        data={
                            impl1_col: type_data[impl1_col],
                            impl2_col: type_data[impl2_col],
                            "diff": type_data[impl2_col] - type_data[impl1_col],
                            "format": [comp_type] * len(type_data),
                            "quality": (
                                type_data["quality"]
                                if "quality" in type_data.columns
                                else [0] * len(type_data)
                            ),
                        }
                    )

                    # Use scatter instead of circle
                    p.scatter(
                        impl1_col,
                        impl2_col,
                        source=type_source,
                        size=10,
                        color=DARK_COLORS[i % len(DARK_COLORS)],
                        alpha=0.7,
                        legend_label=comp_type,
                    )
        else:
            # Default to a nice color if no format information
            source = ColumnDataSource(
                data={
                    impl1_col: plot_data[impl1_col],
                    impl2_col: plot_data[impl2_col],
                    "diff": plot_data[impl2_col] - plot_data[impl1_col],
                    "format": ["Unknown"] * len(plot_data),
                    "quality": (
                        plot_data["quality"]
                        if "quality" in plot_data.columns
                        else [0] * len(plot_data)
                    ),
                }
            )

            p.scatter(
                impl1_col,
                impl2_col,
                source=source,
                size=10,
                color=DARK_COLORS[0],
                alpha=0.7,
            )

        # Fit linear regression model
        model = fit_regression_model(
            plot_data[impl1_col].values,
            plot_data[impl2_col].values,
            model_type="linear",
        )

        if model:
            # Add regression line
            x_range = np.linspace(
                plot_data[impl1_col].min(), plot_data[impl1_col].max(), 100
            )
            p.line(
                x_range,
                model["func"](x_range),
                color="#FF3B30",  # Bright red
                line_width=3,
                alpha=1.0,
                legend_label=f"Regression: {model['formula']} (R² = {model['r2']:.3f})",
            )

            # Add perfect correlation line (y = x)
            p.line(
                [0, 100],
                [0, 100],
                color="white",
                line_width=1.5,
                line_dash="dashed",
                alpha=0.8,
                legend_label="Perfect correlation (y = x)",
            )

        p.legend.location = "top_left"
        p.legend.background_fill_color = "#2a2a2a"
        p.legend.background_fill_alpha = 0.7
        p.legend.border_line_color = "#666666"
        p.legend.label_text_color = "white"
        p.legend.title_text_color = "white"

        # Save the HTML figure
        save(p)
        figure_paths.append(
            os.path.join(
                output_dir,
                f"{impl1_name.lower()}_{impl2_name.lower()}_correlation.html",
            )
        )

    return figure_paths


def _plot_implementation_differences(data, output_dir, formats):
    """Plot distribution of differences between implementations."""
    figure_paths = []

    # Get list of available implementations
    implementations = []
    for impl in ["cpp_score", "python_score", "rust_score"]:
        if impl in data.columns and not data[impl].isna().all():
            implementations.append(impl)

    if len(implementations) < 2:
        print("Warning: Need at least two implementations to plot differences")
        return figure_paths

    # Calculate differences between all implementation pairs
    diffs = {}
    for i, impl1 in enumerate(implementations):
        for impl2 in implementations[i + 1 :]:
            # Get clean implementation names
            impl1_name = impl1.split("_")[0].capitalize()
            impl2_name = impl2.split("_")[0].capitalize()

            # Calculate differences
            # Filter out rows with missing values
            diff_data = data.dropna(subset=[impl1, impl2])
            if len(diff_data) == 0:
                continue

            diff_name = f"{impl2_name} - {impl1_name}"
            diffs[diff_name] = diff_data[impl2] - diff_data[impl1]

    if not diffs:
        print("Warning: No valid implementation differences to plot")
        return figure_paths

    # Create the matplotlib figure
    if "png" in formats:
        plt.figure(figsize=(12, 6))

        # Plot histogram of differences
        for i, (diff_name, diff_values) in enumerate(diffs.items()):
            plt.hist(
                diff_values,
                bins=30,
                alpha=0.7,
                color=DARK_COLORS[i % len(DARK_COLORS)],
                edgecolor="#222222",
                linewidth=0.5,
                label=f"{diff_name} (mean: {diff_values.mean():.3f})",
            )

            # Add vertical line at mean
            plt.axvline(
                diff_values.mean(),
                color=DARK_COLORS[i % len(DARK_COLORS)],
                linestyle="--",
                linewidth=2,
                alpha=0.8,
            )

        plt.xlabel("Difference in SSIMULACRA2 Score")
        plt.ylabel("Frequency")
        plt.title("Distribution of Differences Between SSIMULACRA2 Implementations")
        plt.legend(framealpha=0.7)
        plt.grid(True, alpha=0.4, linestyle="--", color="#444444")
        plt.tight_layout()

        # Save figure
        fig_path = os.path.join(output_dir, "implementation_differences.png")
        plt.savefig(fig_path, bbox_inches="tight")
        plt.close()
        figure_paths.append(fig_path)

    # Create the Bokeh figure
    if "html" in formats and BOKEH_AVAILABLE:
        output_file(os.path.join(output_dir, "implementation_differences.html"))

        p = figure(
            title="Distribution of Differences Between SSIMULACRA2 Implementations",
            x_axis_label="Difference in SSIMULACRA2 Score",
            y_axis_label="Frequency",
            width=900,
            height=500,
            background_fill_color="#121212",
            border_fill_color="#121212",
            outline_line_color="#444444",
        )

        # Add dark theme styling
        p.title.text_color = "white"
        p.xaxis.axis_label_text_color = "white"
        p.yaxis.axis_label_text_color = "white"
        p.xaxis.major_label_text_color = "white"
        p.yaxis.major_label_text_color = "white"
        p.xgrid.grid_line_color = "#333333"
        p.ygrid.grid_line_color = "#333333"

        # Add each difference distribution as histogram
        for i, (diff_name, diff_values) in enumerate(diffs.items()):
            hist, edges = np.histogram(diff_values, bins=30)

            source = ColumnDataSource(
                data=dict(
                    left=edges[:-1],
                    right=edges[1:],
                    top=hist,
                    name=[diff_name] * len(hist),
                    mean=[diff_values.mean()] * len(hist),
                )
            )

            # Add histogram
            p.quad(
                top="top",
                bottom=0,
                left="left",
                right="right",
                source=source,
                fill_color=DARK_COLORS[i % len(DARK_COLORS)],
                line_color="#222222",
                alpha=0.7,
                legend_label=f"{diff_name} (mean: {diff_values.mean():.3f})",
            )

            # Add mean line
            p.line(
                [diff_values.mean(), diff_values.mean()],
                [0, hist.max() * 1.1],
                line_color=DARK_COLORS[i % len(DARK_COLORS)],
                line_width=2.5,
                line_dash="dashed",
                alpha=0.9,
            )

        p.legend.location = "top_right"
        p.legend.background_fill_color = "#2a2a2a"
        p.legend.background_fill_alpha = 0.7
        p.legend.border_line_color = "#666666"
        p.legend.label_text_color = "white"
        p.legend.title_text_color = "white"

        # Save the HTML figure
        save(p)
        figure_paths.append(os.path.join(output_dir, "implementation_differences.html"))

    return figure_paths


def _plot_implementation_by_format(data, output_dir, formats):
    """Plot implementation performance by compression format."""
    figure_paths = []

    # Check if we have the necessary columns
    if "compression_type" not in data.columns:
        print(
            "Warning: Missing compression_type column for implementation by format plot"
        )
        return figure_paths

    # Get list of available implementations
    implementations = []
    for impl in ["cpp_score", "python_score", "rust_score"]:
        if impl in data.columns and not data[impl].isna().all():
            implementations.append(impl)

    if len(implementations) < 2:
        print("Warning: Need at least two implementations to plot by format")
        return figure_paths

    # Get unique compression types
    compression_types = data["compression_type"].unique()

    # Create the matplotlib figure
    if "png" in formats:
        plt.figure(figsize=(12, 8))

        # For each compression type, calculate average score for each implementation
        avg_scores = {}
        for comp_type in compression_types:
            # Use boolean indexing instead of equality
            type_mask = data["compression_type"] == comp_type
            if not type_mask.any():
                continue

            type_data = data[type_mask]
            avg_scores[comp_type] = {}

            for impl in implementations:
                impl_data = type_data.dropna(subset=[impl])
                if len(impl_data) > 0:
                    avg_scores[comp_type][impl] = impl_data[impl].mean()

        # Plot bar chart
        formats_with_data = list(avg_scores.keys())
        x = np.arange(len(formats_with_data))
        width = 0.8 / len(implementations)

        for i, impl in enumerate(implementations):
            # Get clean implementation name
            impl_name = impl.split("_")[0].capitalize()

            # Get values for this implementation
            values = [avg_scores[ct].get(impl, 0) for ct in formats_with_data]

            plt.bar(
                x + (i - len(implementations) / 2 + 0.5) * width,
                values,
                width=width,
                label=impl_name,
                color=DARK_COLORS[i % len(DARK_COLORS)],
                edgecolor="#222222",
                linewidth=0.5,
            )

            # Add value labels on top of bars
            for j, v in enumerate(values):
                plt.text(
                    x[j] + (i - len(implementations) / 2 + 0.5) * width,
                    v + 1,
                    f"{v:.1f}",
                    ha="center",
                    va="bottom",
                    color=DARK_COLORS[i % len(DARK_COLORS)],
                    fontsize=9,
                    rotation=0,
                )

        plt.xlabel("Compression Format")
        plt.ylabel("Average SSIMULACRA2 Score")
        plt.title("Average SSIMULACRA2 Score by Implementation and Format")
        plt.xticks(x, formats_with_data)
        plt.legend(title="Implementation", framealpha=0.7)
        plt.grid(True, alpha=0.4, linestyle="--", color="#444444")
        plt.tight_layout()

        # Save figure
        fig_path = os.path.join(output_dir, "implementation_by_format.png")
        plt.savefig(fig_path, bbox_inches="tight")
        plt.close()
        figure_paths.append(fig_path)

    # Create the Bokeh figure
    if "html" in formats and BOKEH_AVAILABLE:
        output_file(os.path.join(output_dir, "implementation_by_format.html"))

        # Create a filtered list of formats that actually have data
        formats_with_data = []

        for comp_type in compression_types:
            type_mask = data["compression_type"] == comp_type
            if type_mask.any():  # Check if this format actually has data
                # Check if it has implementation data
                type_data = data[type_mask]
                if any(not type_data[impl].isna().all() for impl in implementations):
                    formats_with_data.append(comp_type)

        if not formats_with_data:
            print("Warning: No valid format data for implementation comparison")
            return figure_paths

        p = figure(
            title="Average SSIMULACRA2 Score by Implementation and Format",
            x_range=formats_with_data,
            y_range=(0, 100),
            x_axis_label="Compression Format",
            y_axis_label="Average SSIMULACRA2 Score",
            width=900,
            height=500,
            tooltips=[
                ("Implementation", "@impl"),
                ("Format", "@format"),
                ("Avg Score", "@score{0.00}"),
                ("Sample Size", "@count"),
            ],
            background_fill_color="#121212",
            border_fill_color="#121212",
            outline_line_color="#444444",
        )

        # Add dark theme styling
        p.title.text_color = "white"
        p.xaxis.axis_label_text_color = "white"
        p.yaxis.axis_label_text_color = "white"
        p.xaxis.major_label_text_color = "white"
        p.yaxis.major_label_text_color = "white"
        p.xgrid.grid_line_color = "#333333"
        p.ygrid.grid_line_color = "#333333"

        # For each implementation, add bars
        width = 0.8 / len(implementations)

        for i, impl in enumerate(implementations):
            # Get clean implementation name
            impl_name = impl.split("_")[0].capitalize()

            # Prepare data
            source_data = {"format": [], "score": [], "count": [], "impl": []}

            for j, comp_type in enumerate(formats_with_data):
                type_data = data[data["compression_type"] == comp_type]
                impl_data = type_data.dropna(subset=[impl])

                if len(impl_data) > 0:
                    source_data["format"].append(comp_type)
                    source_data["score"].append(float(impl_data[impl].mean()))
                    source_data["count"].append(len(impl_data))
                    source_data["impl"].append(impl_name)

            if not source_data["format"]:  # Skip if no data for this implementation
                continue

            source = ColumnDataSource(data=source_data)

            # Calculate adjusted x positions
            x_offset = (i - len(implementations) / 2 + 0.5) * width

            # Add bars
            p.vbar(
                x=dodge("format", x_offset, range=p.x_range),
                top="score",
                width=width * 0.9,
                source=source,
                color=DARK_COLORS[i % len(DARK_COLORS)],
                line_color="#222222",
                alpha=0.8,
                legend_label=impl_name,
            )

        p.legend.location = "top_right"
        p.legend.background_fill_color = "#2a2a2a"
        p.legend.background_fill_alpha = 0.7
        p.legend.border_line_color = "#666666"
        p.legend.label_text_color = "white"
        p.legend.title_text_color = "white"
        p.x_range.range_padding = 0.1

        # Save the HTML figure
        save(p)
        figure_paths.append(os.path.join(output_dir, "implementation_by_format.html"))

    return figure_paths
