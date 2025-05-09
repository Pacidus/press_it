"""Core data handling for benchmark analysis."""

import os
from pathlib import Path
import pandas as pd


def find_benchmark_files(results_dir=None):
    """Find all benchmark Parquet files in the results directory.

    Args:
        results_dir: Directory to search in (default: current directory)

    Returns:
        list: Sorted list of benchmark file paths
    """
    search_dir = Path(results_dir or ".")
    benchmark_files = []

    # Look for parquet files with benchmark in the name
    patterns = ["*benchmark*.parquet", "*ssimulacra2*.parquet"]

    for pattern in patterns:
        benchmark_files.extend(search_dir.glob(pattern))

    # Sort by modification time (newest last)
    benchmark_files.sort(key=lambda x: os.path.getmtime(x))
    return benchmark_files


def load_benchmark_data(file_path):
    """Load benchmark data from a Parquet file.

    Args:
        file_path: Path to benchmark Parquet file

    Returns:
        DataFrame: Loaded benchmark data or None if loading fails
    """
    try:
        df = pd.read_parquet(file_path)
        print(f"Loaded {len(df)} rows from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading benchmark data: {e}")
        return None


def combine_benchmark_files(file_paths, output_path=None):
    """Combine multiple benchmark files into a single DataFrame.

    Args:
        file_paths: List of benchmark file paths
        output_path: Path to save combined data

    Returns:
        DataFrame: Combined benchmark data
    """
    combined_df = pd.DataFrame()

    for file_path in file_paths:
        try:
            df = pd.read_parquet(file_path)
            print(f"Adding {len(df)} rows from {file_path}")
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    # Deduplicate if needed
    if len(combined_df) > 0:
        before_count = len(combined_df)
        combined_df = combined_df.drop_duplicates(
            subset=["original_path", "compressed_path"]
        )
        after_count = len(combined_df)

        if before_count > after_count:
            print(f"Removed {before_count - after_count} duplicate entries")

        # Sort by timestamp if available
        if "timestamp" in combined_df.columns:
            combined_df = combined_df.sort_values("timestamp")

        # Save combined data if requested
        if output_path:
            combined_df.to_parquet(
                output_path, engine="pyarrow", compression="snappy", index=False
            )
            print(f"Saved combined data ({len(combined_df)} rows) to {output_path}")

    return combined_df


def add_categorical_columns(df):
    """Add useful categorical columns to the DataFrame.

    Args:
        df: DataFrame with benchmark data

    Returns:
        DataFrame: DataFrame with added columns
    """
    if len(df) == 0:
        return df

    # Make a copy to avoid modifying the original
    result = df.copy()

    # Add image size category
    if "width" in result.columns and "height" in result.columns:
        result["image_size"] = result["width"] * result["height"]
        result["size_category"] = pd.cut(
            result["image_size"],
            bins=[0, 500000, 1000000, 2000000, float("inf")],
            labels=["small", "medium", "large", "very_large"],
        )

    # Add quality bins
    if "python_score" in result.columns:
        result["quality_bin"] = pd.cut(
            result["python_score"],
            bins=[0, 60, 70, 80, 90, 100],
            labels=["0-60", "60-70", "70-80", "80-90", "90-100"],
        )

    return result


def get_best_score_column(df):
    """Get the most available score column to use as reference.

    Args:
        df: DataFrame with benchmark data

    Returns:
        str: Column name for best available score
    """
    # Priority: python, cpp, rust
    if "python_score" in df.columns and df["python_score"].notna().any():
        return "python_score"
    elif "cpp_score" in df.columns and df["cpp_score"].notna().any():
        return "cpp_score"
    elif "rust_score" in df.columns and df["rust_score"].notna().any():
        return "rust_score"
    else:
        raise ValueError("No valid score columns found in the data")
