"""Core data handling for benchmark analysis."""

import os
from pathlib import Path
import pandas as pd
import numpy as np


def find_benchmark_files(results_dir=None):
    """Find all benchmark Parquet files in the results directory.

    Args:
        results_dir: Directory to search in (default: current directory)

    Returns:
        list: Sorted list of benchmark file paths with no duplicates
    """
    search_dir = Path(results_dir or ".")
    benchmark_files = []

    # Look for parquet files with benchmark in the name
    patterns = ["*benchmark*.parquet", "*ssimulacra2*.parquet"]

    for pattern in patterns:
        # Convert to Path objects and get absolute paths to prevent duplicates
        found_files = [file.resolve() for file in search_dir.glob(pattern)]
        benchmark_files.extend(found_files)

    # Remove duplicates by converting to dict keys and back to list
    # This preserves ordering while removing duplicates
    unique_files = list(dict.fromkeys(benchmark_files))

    # Sort by modification time (newest last)
    unique_files.sort(key=lambda x: os.path.getmtime(x))

    return unique_files


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


def optimize_dataframe(df):
    """Optimize a dataframe for storage by using appropriate data types.

    Args:
        df: DataFrame to optimize

    Returns:
        DataFrame: Optimized dataframe
    """
    if len(df) == 0:
        return df

    # Make a copy to avoid modifying the original
    result = df.copy()

    # First convert any categoricals back to strings
    for col in result.columns:
        if pd.api.types.is_categorical_dtype(result[col]):
            result[col] = result[col].astype(str)

    # Then convert string columns to categorical for better compression
    for col in result.columns:
        if pd.api.types.is_object_dtype(result[col]):
            # Check if it contains string data
            if result[col].dropna().map(lambda x: isinstance(x, str)).all():
                result[col] = result[col].astype("category")

    # Optimize numeric columns
    for col in result.columns:
        if pd.api.types.is_numeric_dtype(
            result[col]
        ) and not pd.api.types.is_categorical_dtype(result[col]):
            # Downcast integers if possible
            if pd.api.types.is_integer_dtype(result[col]):
                min_val = result[col].min()
                max_val = result[col].max()

                # Choose smallest possible int type
                if min_val >= 0:
                    if max_val <= 255:
                        result[col] = result[col].astype(np.uint8)
                    elif max_val <= 65535:
                        result[col] = result[col].astype(np.uint16)
                    elif max_val <= 4294967295:
                        result[col] = result[col].astype(np.uint32)
                else:
                    if min_val >= -128 and max_val <= 127:
                        result[col] = result[col].astype(np.int8)
                    elif min_val >= -32768 and max_val <= 32767:
                        result[col] = result[col].astype(np.int16)
                    elif min_val >= -2147483648 and max_val <= 2147483647:
                        result[col] = result[col].astype(np.int32)

    return result


def combine_benchmark_files(file_paths, output_path=None, optimize=True):
    """Combine multiple benchmark files into a single DataFrame.

    Args:
        file_paths: List of benchmark file paths
        output_path: Path to save combined data
        optimize: Whether to optimize the dataframe for storage

    Returns:
        DataFrame: Combined benchmark data
    """
    if not file_paths:
        print("No files to combine")
        return pd.DataFrame()

    # Remove any duplicate file paths
    unique_paths = list(dict.fromkeys([Path(p).resolve() for p in file_paths]))
    if len(unique_paths) < len(file_paths):
        print(
            f"Note: Removed {len(file_paths) - len(unique_paths)} duplicate file references"
        )

    # Load each file as pandas DataFrame
    dataframes = []

    for file_path in unique_paths:
        try:
            df = pd.read_parquet(file_path)

            print(f"Added {len(df)} rows from {file_path}")
            dataframes.append(df)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    if not dataframes:
        print("No valid data loaded")
        return pd.DataFrame()

    # Combine dataframes directly with pandas.concat
    try:
        # Concatenate pandas DataFrames
        if len(dataframes) > 1:
            combined_df = pd.concat(dataframes, ignore_index=True)
        else:
            combined_df = dataframes[0].copy()

        # Identify key columns for deduplication
        critical_columns = [
            "width",
            "height",
            "original_size",
            "compressed_size",
            "compression_type",
            "quality",
        ]

        # Add any score columns that exist for better deduplication
        score_columns = [col for col in combined_df.columns if col.endswith("_score")]
        dedup_columns = critical_columns + score_columns

        # Ensure all dedup columns exist
        dedup_columns = [col for col in dedup_columns if col in combined_df.columns]

        # Remove duplicates
        before_count = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=dedup_columns)
        after_count = len(combined_df)

        if before_count > after_count:
            print(f"Removed {before_count - after_count} duplicate entries")

        # Save with optimal settings
        if output_path:
            # Apply type optimizations for better storage efficiency
            if optimize:
                combined_df = optimize_dataframe(combined_df)

            # Save with optimal settings
            combined_df.to_parquet(
                output_path,
                engine="pyarrow",
                compression="zstd",
                compression_level=9,
                index=False,
                use_dictionary=True,  # Always use dictionary encoding
                coerce_timestamps="ms",
                allow_truncated_timestamps=True,
            )

            print(f"Saved combined data ({len(combined_df)} rows) to {output_path}")

        return combined_df

    except Exception as e:
        print(f"Error combining files: {e}")
        import traceback

        traceback.print_exc()
        return pd.DataFrame()


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


def calculate_compression_ratio(df):
    """Calculate compression ratio for a DataFrame.

    Args:
        df: DataFrame with benchmark data

    Returns:
        Series: Compression ratio for each row
    """
    return df.apply(
        lambda row: (
            row["original_size"] / row["compressed_size"]
            if row["compressed_size"] > 0
            else float("inf")
        ),
        axis=1,
    )
