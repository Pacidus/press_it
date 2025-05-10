"""CLI interface for press_it benchmark visualization."""

import argparse
import sys
import os
import pandas as pd
from pathlib import Path

from press_it import __version__
from press_it.benchmark.analysis.visualization import (
    format_comparison_plots,
    implementation_comparison_plots,
)
from press_it.cli.main import create_parent_parser, run_with_args
from press_it.utils.subprocess_utils import check_dependencies


# Optional dependencies for visualization
RECOMMENDED_DEPENDENCIES = [
    "python3",  # Just to ensure Python is available
]


def parse_visualization_args():
    """Parse command-line arguments for visualization.

    Returns:
        Namespace: Parsed arguments
    """
    parent_parser = create_parent_parser()

    parser = argparse.ArgumentParser(
        description="Visualize benchmark data from press_it",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[parent_parser],
        epilog=parent_parser.epilog,
    )

    parser.add_argument("parquet_file", type=str, help="Path to benchmark Parquet file")

    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="figures",
        help="Output directory for generated figures",
    )

    parser.add_argument(
        "--format",
        "-f",
        type=str,
        default="all",
        choices=["png", "html", "all"],
        help="Output format for figures (png, html, or all)",
    )

    parser.add_argument(
        "--analysis-type",
        "-a",
        type=str,
        default="all",
        choices=["formats", "implementations", "all"],
        help="Type of analysis to visualize",
    )

    parser.add_argument("--dpi", type=int, default=300, help="DPI for PNG figures")

    # Set version string to match main CLI
    parser.add_argument(
        "--version", "-V", action="version", version=f"press-viz {__version__}"
    )

    return parser.parse_args()


def run_visualization(args):
    """Run the visualization with provided arguments.

    Args:
        args: Parsed command-line arguments

    Returns:
        int: Exit code (0 for success, non-zero for error)
    """
    try:
        # Check if file exists
        if not os.path.isfile(args.parquet_file):
            print(
                f"Error: Parquet file not found: {args.parquet_file}", file=sys.stderr
            )
            return 1

        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)

        # Load the benchmark data
        try:
            df = pd.read_parquet(args.parquet_file)
            print(f"Loaded {len(df)} rows from {args.parquet_file}")
        except Exception as e:
            print(f"Error loading Parquet file: {e}", file=sys.stderr)
            return 1

        # Empty dataframe check
        if len(df) == 0:
            print("Error: Parquet file contains no data", file=sys.stderr)
            return 1

        # Determine output formats
        formats = []
        if args.format == "all":
            formats = ["png", "html"]
        else:
            formats = [args.format]

        # Check if bokeh is available for HTML output
        if "html" in formats:
            try:
                import bokeh

                print(f"Using bokeh {bokeh.__version__} for HTML visualizations")
            except ImportError:
                print("Warning: bokeh not installed, falling back to PNG only")
                formats = ["png"]

        # Set matplotlib DPI for PNG output
        if "png" in formats:
            import matplotlib as mpl

            mpl.rcParams["savefig.dpi"] = args.dpi
            print(f"Setting PNG resolution to {args.dpi} DPI")

        # Generate visualizations based on analysis type
        figure_paths = []

        if args.analysis_type in ["formats", "all"]:
            print("\nGenerating format comparison visualizations...")
            format_figures = format_comparison_plots(df, args.output_dir, formats)
            figure_paths.extend(format_figures)
            print(f"Created {len(format_figures)} format comparison figures")

        if args.analysis_type in ["implementations", "all"]:
            print("\nGenerating implementation comparison visualizations...")
            impl_figures = implementation_comparison_plots(df, args.output_dir, formats)
            figure_paths.extend(impl_figures)
            print(f"Created {len(impl_figures)} implementation comparison figures")

        # Summary
        if figure_paths:
            print(
                f"\nSuccessfully created {len(figure_paths)} visualization files in {args.output_dir}/"
            )

            # List PNG files (for README)
            png_files = [p for p in figure_paths if p.endswith(".png")]
            if png_files:
                print("\nPNG files for README:")
                for png in png_files:
                    rel_path = os.path.relpath(png, os.getcwd())
                    print(f"![{os.path.basename(png)}]({rel_path})")

            # List HTML files (for website)
            html_files = [p for p in figure_paths if p.endswith(".html")]
            if html_files:
                print("\nHTML files for website:")
                for html in html_files:
                    print(f"- {os.path.relpath(html, os.getcwd())}")

            return 0
        else:
            print("No visualizations were generated")
            return 1

    except Exception as e:
        print(f"Error during visualization: {e}", file=sys.stderr)
        # Print traceback for debugging
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def main():
    """Main entry point for the visualization CLI."""
    # Create and parse arguments
    args = parse_visualization_args()

    # Optional dependency check (we don't require specific visualization tools)
    try:
        check_dependencies(RECOMMENDED_DEPENDENCIES)
    except Exception as e:
        # Only warn, don't exit
        print(f"Warning: {e}", file=sys.stderr)

    # Run visualization
    return run_visualization(args)


if __name__ == "__main__":
    sys.exit(main())
