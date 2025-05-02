#!/usr/bin/env python3
"""
SSIMULACRA2 Benchmark Tool with Multiple Compression Formats

This script:
1. Downloads random images from Picsum Photos
2. Compresses them using multiple formats (MozJPEG, WebP, AVIF)
3. Evaluates the compressed images with three SSIMULACRA2 implementations
4. Runs continuously until interrupted with Ctrl+C
5. Analyzes and saves the collected data upon termination
"""

import os
import sys
import time
import random
import subprocess
import signal
import json
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from scipy import stats
from datetime import datetime
import tempfile
import shutil

# Import your Python implementation of SSIMULACRA2
try:
    from ssimulacra2 import compute_ssimulacra2_with_alpha
except ImportError:
    print("Warning: Could not import ssimulacra2 module. Python implementation will be skipped.")
    compute_ssimulacra2_with_alpha = None

# Configuration
TEMP_DIR = Path("./benchmark_temp")
RESULTS_DIR = Path("./benchmark_results")
ORIGINAL_IMAGE_DIR = TEMP_DIR / "originals"
COMPRESSED_IMAGE_DIR = TEMP_DIR / "compressed"
DECODED_DIR = TEMP_DIR / "decoded"
OUTPUT_FILE = RESULTS_DIR / f"ssimulacra2_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
ANALYSIS_FILE = RESULTS_DIR / f"ssimulacra2_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
PLOTS_DIR = RESULTS_DIR / "plots"

# Global variables
results = []
running = True

# Check for required dependencies
DEPENDENCIES = [
    "magick",
    "cjpeg",
    "cwebp",
    "dwebp",
    "avifenc",
    "avifdec",
]

# File extensions mapping
FILE_EXTENSIONS = {
    "mozjpeg": "jpg",
    "webp": "webp",
    "avif": "avif",
}

def check_dependencies(required_tools):
    """Verify required system utilities are available."""
    missing = []
    for tool in required_tools:
        if not shutil.which(tool):
            missing.append(tool)

    if missing:
        print(f"Missing required tools: {', '.join(missing)}. "
              "Please install them before running this script.")
        sys.exit(1)

def setup_directories():
    """Create necessary directories if they don't exist."""
    for dir_path in [TEMP_DIR, RESULTS_DIR, ORIGINAL_IMAGE_DIR, COMPRESSED_IMAGE_DIR, DECODED_DIR, PLOTS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

def signal_handler(sig, frame):
    """Handle Ctrl+C to gracefully stop the script."""
    global running
    print("\nStopping... Please wait for current operation to complete and data to be analyzed.")
    running = False

def get_random_image():
    """Download a random image from Picsum Photos with random dimensions."""
    image_id = f"image_{len(results):04d}"
    output_path = ORIGINAL_IMAGE_DIR / f"{image_id}.png"
    
    # Skip download if file already exists (from a previous run)
    if output_path.exists():
        return str(output_path)
    
    try:
        # Get a random image with random dimensions
        width = random.choice([800, 1024, 1280, 1600])
        height = random.choice([600, 768, 960, 1200])
        
        # Simple URL for random image with specified dimensions
        image_url = f"https://picsum.photos/{width}/{height}"
        
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        
        # Save the image as PNG using ImageMagick for consistency
        temp_jpg = TEMP_DIR / f"temp_{image_id}.jpg"
        with open(temp_jpg, "wb") as f:
            f.write(response.content)
        
        # Convert to PNG with alpha removed for consistent comparison
        subprocess.run(
            ["magick", str(temp_jpg), "-alpha", "off", str(output_path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # Remove temporary file
        os.remove(temp_jpg)
        
        print(f"Downloaded new image: {output_path} ({width}x{height})")
        return str(output_path)
    
    except Exception as e:
        print(f"Error downloading image: {e}")
        
        # If download fails, use a fallback local image if available
        fallback_images = list(ORIGINAL_IMAGE_DIR.glob("*.png"))
        if fallback_images:
            fallback_path = random.choice(fallback_images)
            print(f"Using fallback image: {fallback_path}")
            return str(fallback_path)
        
        # If no fallback available, raise the exception
        raise

def encode_mozjpeg(input_png, quality=None):
    """Encode image to JPEG using MozJPEG."""
    img_path = Path(input_png)
    image_id = img_path.stem
    
    # If quality is not provided, select randomly
    if quality is None:
        quality = random.randint(5, 95)
    
    output_path = COMPRESSED_IMAGE_DIR / f"{image_id}_mozjpeg_{quality}.jpg"
    decoded_png = DECODED_DIR / f"{image_id}_mozjpeg_{quality}_decoded.png"
    
    # Skip compression if file already exists (from a previous run)
    if output_path.exists() and decoded_png.exists():
        return str(output_path), str(decoded_png), "mozjpeg", quality
    
    try:
        # Compress using MozJPEG (based on your example code)
        subprocess.run(
            [
                "cjpeg",
                "-quality",
                str(quality),
                "-outfile",
                str(output_path),
                input_png,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        
        # Decode back to PNG for consistent comparison
        subprocess.run(
            ["magick", str(output_path), str(decoded_png)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        
        print(f"Compressed image: {output_path} (mozjpeg, quality={quality})")
        return str(output_path), str(decoded_png), "mozjpeg", quality
    
    except Exception as e:
        print(f"Error compressing image with MozJPEG: {e}")
        raise

def encode_webp(input_png, quality=None):
    """Encode image to WebP format using cwebp."""
    img_path = Path(input_png)
    image_id = img_path.stem
    
    # If quality is not provided, select randomly
    if quality is None:
        quality = random.randint(5, 95)
    
    output_path = COMPRESSED_IMAGE_DIR / f"{image_id}_webp_{quality}.webp"
    decoded_png = DECODED_DIR / f"{image_id}_webp_{quality}_decoded.png"
    
    # Skip compression if file already exists (from a previous run)
    if output_path.exists() and decoded_png.exists():
        return str(output_path), str(decoded_png), "webp", quality
    
    try:
        # Compress using cwebp (based on your example code)
        subprocess.run(
            ["cwebp", "-m", "6", "-q", str(quality), input_png, "-o", str(output_path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        
        # Decode back to PNG for consistent comparison
        subprocess.run(
            ["dwebp", str(output_path), "-o", str(decoded_png)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        
        print(f"Compressed image: {output_path} (webp, quality={quality})")
        return str(output_path), str(decoded_png), "webp", quality
    
    except Exception as e:
        print(f"Error compressing image with WebP: {e}")
        raise

def encode_avif(input_png, quality=None):
    """Encode image to AVIF format using avifenc."""
    img_path = Path(input_png)
    image_id = img_path.stem
    
    # If quality is not provided, select randomly
    if quality is None:
        quality = random.randint(5, 95)
    
    output_path = COMPRESSED_IMAGE_DIR / f"{image_id}_avif_{quality}.avif"
    decoded_png = DECODED_DIR / f"{image_id}_avif_{quality}_decoded.png"
    
    # Skip compression if file already exists (from a previous run)
    if output_path.exists() and decoded_png.exists():
        return str(output_path), str(decoded_png), "avif", quality
    
    try:
        # Compress using avifenc (based on your example code)
        subprocess.run(
            ["avifenc", "-q", str(quality), "-j", "all", "-s", "0", input_png, str(output_path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        
        # Decode back to PNG for consistent comparison
        subprocess.run(
            ["avifdec", str(output_path), str(decoded_png)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        
        print(f"Compressed image: {output_path} (avif, quality={quality})")
        return str(output_path), str(decoded_png), "avif", quality
    
    except Exception as e:
        print(f"Error compressing image with AVIF: {e}")
        raise

def compress_image(original_path):
    """Compress the image using a random compression method and quality level."""
    # Choose a random encoder
    encoder_funcs = [encode_mozjpeg, encode_webp, encode_avif]
    encoder = random.choice(encoder_funcs)
    
    # Apply random quality
    quality = random.randint(5, 95)
    
    try:
        return encoder(original_path, quality)
    except Exception as e:
        print(f"Compression failed, trying MozJPEG as fallback: {e}")
        # Fallback to MozJPEG if the chosen encoder fails
        return encode_mozjpeg(original_path, quality)

def run_cpp_ssimulacra2(original_path, compressed_path):
    """Run the C++ implementation of SSIMULACRA2."""
    try:
        cmd = ["cmulacra2", original_path, compressed_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        print(f"Error running C++ implementation: {e}")
        print(f"stderr: {e.stderr}")
        return None
    except Exception as e:
        print(f"Unexpected error with C++ implementation: {e}")
        return None

def run_rust_ssimulacra2(original_path, compressed_path):
    """Run the Rust implementation (as2c) of SSIMULACRA2."""
    try:
        cmd = ["as2c", original_path, compressed_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        print(f"Error running Rust implementation: {e}")
        print(f"stderr: {e.stderr}")
        return None
    except Exception as e:
        print(f"Unexpected error with Rust implementation: {e}")
        return None

def evaluate_image(original_path, decoded_path, compressed_path):
    """Evaluate the image using all three SSIMULACRA2 implementations."""
    print(f"Evaluating: {Path(compressed_path).name}")
    
    # Get file size information
    original_size = os.path.getsize(original_path)
    compressed_size = os.path.getsize(compressed_path)
    compression_ratio = original_size / compressed_size if compressed_size > 0 else float('inf')
    
    # Run evaluations
    python_score = None
    cpp_score = None
    rust_score = None
    
    # Python implementation
    if compute_ssimulacra2_with_alpha is not None:
        try:
            python_score = compute_ssimulacra2_with_alpha(original_path, decoded_path)
            print(f"  Python score: {python_score:.4f}")
        except Exception as e:
            print(f"Error with Python implementation: {e}")
    
    # C++ implementation
    try:
        cpp_score = run_cpp_ssimulacra2(original_path, decoded_path)
        print(f"  C++ score: {cpp_score:.4f}")
    except Exception as e:
        print(f"Error with C++ implementation: {e}")
    
    # Rust implementation
    try:
        rust_score = run_rust_ssimulacra2(original_path, decoded_path)
        print(f"  Rust score: {rust_score:.4f}")
    except Exception as e:
        print(f"Error with Rust implementation: {e}")
    
    return {
        "original_path": original_path,
        "decoded_path": decoded_path,
        "compressed_path": compressed_path,
        "original_size": original_size,
        "compressed_size": compressed_size,
        "compression_ratio": compression_ratio,
        "python_score": python_score,
        "cpp_score": cpp_score,
        "rust_score": rust_score
    }

def calculate_corrected_scores(df):
    """Calculate corrected scores based on regression formulas."""
    # If Python scores are available, calculate Python->C++ correction
    if 'python_score' in df.columns and not df['python_score'].isna().all():
        # Use formula from regression analysis
        df['python_corrected'] = 0.931573 * df['python_score'] + 6.347920
    
    # If Rust scores are available, calculate Rust->C++ correction
    if 'rust_score' in df.columns and not df['rust_score'].isna().all():
        # Use formula from regression analysis
        df['rust_corrected'] = 0.898060 * df['rust_score'] + 8.902188
    
    return df

def analyze_results(results_df):
    """Analyze the collected results."""
    print("\nAnalyzing results...")
    
    # Skip analysis if no results or only one result
    if len(results_df) <= 1:
        print("Not enough data for analysis.")
        return {}
    
    # Calculate corrected scores
    results_df = calculate_corrected_scores(results_df)
    
    # Filter out rows with missing values for correlation analysis
    correlation_df = results_df.dropna(subset=['python_score', 'cpp_score', 'rust_score'])
    
    analysis = {
        "summary": {
            "total_images": len(results_df),
            "compression_methods": results_df['compression_type'].value_counts().to_dict(),
            "avg_compression_ratio": results_df['compression_ratio'].mean()
        },
        "score_ranges": {
            "python": {
                "min": results_df['python_score'].min(),
                "max": results_df['python_score'].max(),
                "mean": results_df['python_score'].mean(),
                "median": results_df['python_score'].median()
            },
            "cpp": {
                "min": results_df['cpp_score'].min(),
                "max": results_df['cpp_score'].max(),
                "mean": results_df['cpp_score'].mean(),
                "median": results_df['cpp_score'].median()
            },
            "rust": {
                "min": results_df['rust_score'].min(),
                "max": results_df['rust_score'].max(),
                "mean": results_df['rust_score'].mean(),
                "median": results_df['rust_score'].median()
            }
        }
    }
    
    # Calculate correlation coefficients if we have enough data
    if len(correlation_df) > 1:
        py_cpp_corr = correlation_df['python_score'].corr(correlation_df['cpp_score'])
        py_rust_corr = correlation_df['python_score'].corr(correlation_df['rust_score'])
        cpp_rust_corr = correlation_df['cpp_score'].corr(correlation_df['rust_score'])
        
        analysis["correlations"] = {
            "python_cpp": py_cpp_corr,
            "python_rust": py_rust_corr,
            "cpp_rust": cpp_rust_corr
        }
        
        # Regression analysis
        if len(correlation_df) > 2:
            # Python -> C++
            py_slope, py_intercept, py_r, _, _ = stats.linregress(
                correlation_df['python_score'], correlation_df['cpp_score']
            )
            
            # Rust -> C++
            rust_slope, rust_intercept, rust_r, _, _ = stats.linregress(
                correlation_df['rust_score'], correlation_df['cpp_score']
            )
            
            analysis["regression"] = {
                "python_to_cpp": {
                    "slope": py_slope,
                    "intercept": py_intercept,
                    "r_squared": py_r ** 2,
                    "formula": f"cpp_score = {py_slope:.6f} * python_score + {py_intercept:.6f}"
                },
                "rust_to_cpp": {
                    "slope": rust_slope,
                    "intercept": rust_intercept,
                    "r_squared": rust_r ** 2,
                    "formula": f"cpp_score = {rust_slope:.6f} * rust_score + {rust_intercept:.6f}"
                }
            }
            
            # Calculate mean absolute error of corrected scores
            if 'python_corrected' in results_df.columns:
                py_mae = abs(results_df['cpp_score'] - results_df['python_corrected']).mean()
                analysis["correction_errors"] = {"python_mae": py_mae}
            
            if 'rust_corrected' in results_df.columns:
                rust_mae = abs(results_df['cpp_score'] - results_df['rust_corrected']).mean()
                if "correction_errors" not in analysis:
                    analysis["correction_errors"] = {}
                analysis["correction_errors"]["rust_mae"] = rust_mae
    
    # Break down analysis by compression type
    compression_types = results_df['compression_type'].unique()
    analysis["by_compression_type"] = {}
    
    for comp_type in compression_types:
        type_df = results_df[results_df['compression_type'] == comp_type]
        type_corr_df = type_df.dropna(subset=['python_score', 'cpp_score', 'rust_score'])
        
        type_analysis = {
            "count": len(type_df),
            "avg_quality": type_df['quality'].mean(),
            "avg_compression_ratio": type_df['compression_ratio'].mean(),
            "score_means": {
                "python": type_df['python_score'].mean(),
                "cpp": type_df['cpp_score'].mean(),
                "rust": type_df['rust_score'].mean()
            }
        }
        
        if len(type_corr_df) > 2:
            # Calculate correlations for this compression type
            py_cpp_corr = type_corr_df['python_score'].corr(type_corr_df['cpp_score'])
            type_analysis["python_cpp_correlation"] = py_cpp_corr
            
            # Python -> C++ regression for this compression type
            py_slope, py_intercept, py_r, _, _ = stats.linregress(
                type_corr_df['python_score'], type_corr_df['cpp_score']
            )
            type_analysis["python_to_cpp_formula"] = f"cpp_score = {py_slope:.6f} * python_score + {py_intercept:.6f}"
            type_analysis["python_to_cpp_r_squared"] = py_r ** 2
            
            # Calculate mean differences
            py_cpp_diff = (type_corr_df['python_score'] - type_corr_df['cpp_score']).mean()
            rust_cpp_diff = (type_corr_df['rust_score'] - type_corr_df['cpp_score']).mean()
            type_analysis["avg_differences"] = {
                "python_cpp": py_cpp_diff,
                "rust_cpp": rust_cpp_diff
            }
        
        analysis["by_compression_type"][comp_type] = type_analysis
    
    # Create visualizations
    create_visualizations(results_df)
    
    return analysis

def create_visualizations(df):
    """Create visualization plots from the results."""
    # Skip if not enough data
    if len(df) <= 1:
        return
    
    # 1. Scatter plot of scores
    plt.figure(figsize=(10, 8))
    valid_scores = df.dropna(subset=['cpp_score'])
    
    if 'python_score' in df.columns and not df['python_score'].isna().all():
        plt.scatter(valid_scores['cpp_score'], valid_scores['python_score'], 
                   label='Python', alpha=0.7)
    
    if 'rust_score' in df.columns and not df['rust_score'].isna().all():
        plt.scatter(valid_scores['cpp_score'], valid_scores['rust_score'], 
                   label='Rust', alpha=0.7)
    
    # Add identity line
    score_cols = [col for col in ['python_score', 'cpp_score', 'rust_score'] 
                 if col in df.columns and not df[col].isna().all()]
    min_score = df[score_cols].min().min()
    max_score = df[score_cols].max().max()
    plt.plot([min_score, max_score], [min_score, max_score], 'k--', alpha=0.5, label='y=x')
    
    plt.xlabel('C++ Score (Reference)')
    plt.ylabel('Implementation Score')
    plt.title('SSIMULACRA2 Implementation Comparison')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'score_comparison.png')
    plt.close()
    
    # 2. Score vs Compression Ratio
    plt.figure(figsize=(10, 8))
    
    comp_types = df['compression_type'].unique()
    markers = {'mozjpeg': 'o', 'webp': 's', 'avif': '^'}
    
    for ctype in comp_types:
        type_df = df[df['compression_type'] == ctype]
        plt.scatter(type_df['compression_ratio'], type_df['cpp_score'], 
                   label=f'C++ ({ctype})', alpha=0.7, marker=markers.get(ctype, 'o'))
    
    plt.xscale('log')
    plt.xlabel('Compression Ratio (log scale)')
    plt.ylabel('SSIMULACRA2 Score (C++)')
    plt.title('Score vs Compression Ratio by Format')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'score_vs_compression.png')
    plt.close()
    
    # 3. Difference plots
    if 'python_score' in df.columns and not df['python_score'].isna().all():
        plt.figure(figsize=(10, 6))
        valid_py = df.dropna(subset=['python_score', 'cpp_score'])
        
        for ctype in comp_types:
            type_df = valid_py[valid_py['compression_type'] == ctype]
            if len(type_df) > 0:
                plt.scatter(type_df['cpp_score'], type_df['python_score'] - type_df['cpp_score'],
                          label=ctype, alpha=0.7, marker=markers.get(ctype, 'o'))
        
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.5)
        plt.xlabel('C++ Score')
        plt.ylabel('Python - C++ Difference')
        plt.title('Python Implementation Difference from C++')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'python_difference.png')
        plt.close()
    
    if 'rust_score' in df.columns and not df['rust_score'].isna().all():
        plt.figure(figsize=(10, 6))
        valid_rust = df.dropna(subset=['rust_score', 'cpp_score'])
        
        for ctype in comp_types:
            type_df = valid_rust[valid_rust['compression_type'] == ctype]
            if len(type_df) > 0:
                plt.scatter(type_df['cpp_score'], type_df['rust_score'] - type_df['cpp_score'],
                          label=ctype, alpha=0.7, marker=markers.get(ctype, 'o'))
        
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.5)
        plt.xlabel('C++ Score')
        plt.ylabel('Rust - C++ Difference')
        plt.title('Rust Implementation Difference from C++')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'rust_difference.png')
        plt.close()
    
    # 4. Histogram of scores by compression type
    plt.figure(figsize=(12, 6))
    
    for ctype in comp_types:
        type_df = df[df['compression_type'] == ctype]
        if len(type_df) > 0:
            plt.hist(type_df['cpp_score'], bins=15, alpha=0.5, label=ctype)
    
    plt.xlabel('C++ SSIMULACRA2 Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Scores by Compression Type')
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'score_distribution_by_type.png')
    plt.close()
    
    # 5. Scores by quality for each compression type
    plt.figure(figsize=(15, 8))
    
    for i, ctype in enumerate(comp_types):
        plt.subplot(1, len(comp_types), i+1)
        type_df = df[df['compression_type'] == ctype]
        
        if len(type_df) > 0:
            plt.scatter(type_df['quality'], type_df['cpp_score'], label='C++', alpha=0.7, color='red')
            
            if 'python_score' in df.columns and not df['python_score'].isna().all():
                plt.scatter(type_df['quality'], type_df['python_score'], label='Python', alpha=0.7, color='blue')
            
            if 'rust_score' in df.columns and not df['rust_score'].isna().all():
                plt.scatter(type_df['quality'], type_df['rust_score'], label='Rust', alpha=0.7, color='green')
        
        plt.xlabel('Quality Setting')
        plt.ylabel('SSIMULACRA2 Score')
        plt.title(f'{ctype.upper()} Quality vs Score')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'quality_vs_score_by_type.png')
    plt.close()
    
    # 6. Combined analysis of differences
    plt.figure(figsize=(12, 8))
    
    # Creating a combined plot of differences vs C++ score, colored by compression type
    if 'python_score' in df.columns and not df['python_score'].isna().all():
        valid_py = df.dropna(subset=['python_score', 'cpp_score'])
        plt.subplot(2, 1, 1)
        
        for ctype in comp_types:
            type_df = valid_py[valid_py['compression_type'] == ctype]
            if len(type_df) > 0:
                plt.scatter(type_df['cpp_score'], type_df['python_score'] - type_df['cpp_score'],
                          label=ctype, alpha=0.7, marker=markers.get(ctype, 'o'))
        
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.5)
        plt.xlabel('C++ Score')
        plt.ylabel('Python - C++ Difference')
        plt.title('Python vs C++ Difference by Format')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    if 'rust_score' in df.columns and not df['rust_score'].isna().all():
        valid_rust = df.dropna(subset=['rust_score', 'cpp_score'])
        plt.subplot(2, 1, 2)
        
        for ctype in comp_types:
            type_df = valid_rust[valid_rust['compression_type'] == ctype]
            if len(type_df) > 0:
                plt.scatter(type_df['cpp_score'], type_df['rust_score'] - type_df['cpp_score'],
                          label=ctype, alpha=0.7, marker=markers.get(ctype, 'o'))
        
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.5)
        plt.xlabel('C++ Score')
        plt.ylabel('Rust - C++ Difference')
        plt.title('Rust vs C++ Difference by Format')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'combined_differences.png')
    plt.close()

def save_results(results_data, analysis):
    """Save the collected results and analysis to files."""
    # Create DataFrame from results
    df = pd.DataFrame(results_data)
    
    # Save to CSV
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Results saved to {OUTPUT_FILE}")
    
    # Save analysis to JSON
    with open(ANALYSIS_FILE, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"Analysis saved to {ANALYSIS_FILE}")

def main():
    """Main function to run the benchmark."""
    global results, running
    
    # Setup
    check_dependencies(DEPENDENCIES)
    setup_directories()
    signal.signal(signal.SIGINT, signal_handler)
    
    print("Starting SSIMULACRA2 benchmark. Press Ctrl+C to stop and save results.")
    print(f"Images will be saved to: {ORIGINAL_IMAGE_DIR}")
    print(f"Compressed versions will be saved to: {COMPRESSED_IMAGE_DIR}")
    print(f"Decoded images will be saved to: {DECODED_DIR}")
    
    try:
        while running:
            try:
                # Get a random original image
                original_path = get_random_image()
                
                # Compress with random format and quality
                compressed_path, decoded_path, compression_type, quality = compress_image(original_path)
                
                # Evaluate with all implementations
                result = evaluate_image(original_path, decoded_path, compressed_path)
                result['compression_type'] = compression_type
                result['quality'] = quality
                results.append(result)
                
                # Print a separator
                print("-" * 40)
                
                # Optional small delay to prevent system overload
                time.sleep(0.5)
            
            except KeyboardInterrupt:
                break
            
            except Exception as e:
                print(f"Error in benchmark cycle: {e}")
                time.sleep(1)  # Wait a bit before retrying
    
    finally:
        # Even if there's an unhandled exception, we want to save the results
        if results:
            # Convert to DataFrame
            results_df = pd.DataFrame(results)
            
            # Run analysis
            analysis = analyze_results(results_df)
            
            # Save results
            save_results(results, analysis)
            
            print(f"\nBenchmark complete. Processed {len(results)} images.")
            print(f"Results saved to {OUTPUT_FILE}")
            print(f"Analysis saved to {ANALYSIS_FILE}")
            print(f"Visualizations saved to {PLOTS_DIR}")
        else:
            print("\nNo results collected.")

if __name__ == "__main__":
    main()
