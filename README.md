# press_it

[![PyPI - Version](https://img.shields.io/pypi/v/press-it.svg)](https://pypi.org/project/press-it)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/press-it.svg)](https://pypi.org/project/press-it)
[![License](https://img.shields.io/badge/License-GPL_v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

A tool to compress images with formats targeting a specific perceptual quality using SSIMULACRA2 metrics.

## Overview

press_it intelligently selects the optimal compression format and settings to achieve your desired perceptual quality level. It analyzes the image characteristics and automatically chooses between JPEG, WebP, AVIF, and PNG to deliver the smallest file size while meeting your quality target.

## Features

- **Automatic format selection**: Compares multiple formats (JPEG, WebP, AVIF, PNG) and chooses the optimal one
- **Quality-driven approach**: Set your target quality and let the tool find the best compression parameters
- **SSIMULACRA2 metrics**: Uses perceptual quality measurements that align with human vision
- **Binary search optimization**: Efficiently finds the optimal compression parameters
- **Comprehensive benchmarking**: Includes a benchmark tool to compare formats across quality levels

## Installation

### Prerequisites

The following tools need to be installed on your system:

- ImageMagick (`magick`)
- MozJPEG (`cjpeg`)
- WebP tools (`cwebp`, `dwebp`)
- AVIF tools (`avifenc`, `avifdec`)
- PNG optimization tools (`pngcrush`)

After installing prerequisites, install press_it:

```bash
pip install press-it
```

## Usage

### Image Compression

```bash
# Basic compression targeting 85.0 quality score
press-it input.png 85.0

# Resize during compression
press-it input.png 85.0 --resize 800x600

# Specify output directory
press-it input.png 85.0 --output /path/to/output

# Show detailed information during compression
press-it input.png 85.0 --verbose

# Show version information
press-it --version
```

#### Options

```
usage: press-it [-h] [--version] [--resize RESIZE] [--output OUTPUT] [--verbose] input_image target_ssim

positional arguments:
  input_image           Path to source image file
  target_ssim           Target SSIM value (0-100)

optional arguments:
  -h, --help            show this help message and exit
  --version, -V         show program's version number and exit
  --resize RESIZE, -r RESIZE
                        Resize the image (width)x(height). If one dimension is not set, the aspect ratio is respected
  --output OUTPUT, -o OUTPUT
                        Output directory. If not provided, uses current directory
  --verbose, -v         Show detailed progress information during compression
```

## Benchmarking Tool

The library includes a comprehensive benchmarking tool to evaluate and compare different image formats across various quality settings. This is valuable for:

- Comparing compression efficiency of different formats (JPEG, WebP, AVIF)
- Validating quality vs. file size tradeoffs
- Testing SSIMULACRA2 implementations (Python, C++, Rust)
- Generating data for compression strategy optimization

### Running a Benchmark

```bash
# Basic benchmark with 10 images
press-benchmark --num-images 10 --output benchmark-results.csv

# Extended quality range benchmark
press-benchmark --num-images 20 --quality-min 20 --quality-max 90

# Continuous benchmark (runs until stopped with Ctrl+C)
press-benchmark --temp-dir ./my_benchmark_files

# Show detailed information during benchmark
press-benchmark --num-images 10 --verbose

# Show version information
press-benchmark --version
```

The benchmark process:
1. Downloads random test images from Picsum Photos
2. Processes each image with randomly selected formats (MozJPEG, WebP, AVIF)
3. Applies random quality settings within your specified range
4. Evaluates compressed images using multiple SSIMULACRA2 implementations
5. Records metrics including file size, compression ratio, and quality scores

### Benchmark Options

```
usage: press-benchmark [-h] [--num-images NUM_IMAGES] [--output OUTPUT] [--temp-dir TEMP_DIR] 
                      [--quality-min QUALITY_MIN] [--quality-max QUALITY_MAX] [--verbose] [--version]

optional arguments:
  -h, --help            show this help message and exit
  --num-images NUM_IMAGES, -n NUM_IMAGES
                        Number of images to process (0 for infinite until Ctrl+C) (default: 0)
  --output OUTPUT, -o OUTPUT
                        Output CSV file path (default: auto-generated in benchmark_results directory)
  --temp-dir TEMP_DIR, -t TEMP_DIR
                        Directory for temporary files during benchmark (default: ./benchmark_temp)
  --quality-min QUALITY_MIN
                        Minimum quality value to test (5-100) (default: 5)
  --quality-max QUALITY_MAX
                        Maximum quality value to test (5-100) (default: 95)
  --verbose, -v         Show detailed progress information during benchmark
  --version, -V         show program's version number and exit
```

### Benchmark Results

The benchmark generates a CSV file with detailed information for each test image:

| Column | Description |
|--------|-------------|
| original_path | Path to the original test image |
| compressed_path | Path to the compressed image |
| decoded_path | Path to the decoded image used for evaluation |
| width/height | Image dimensions |
| original_size | Size of the original image in bytes |
| compressed_size | Size of the compressed image in bytes |
| compression_ratio | Ratio of original to compressed size |
| compression_type | Format used (mozjpeg, webp, avif) |
| quality | Compression quality setting used |
| python_score | SSIMULACRA2 score from Python implementation |
| cpp_score | SSIMULACRA2 score from C++ implementation (if available) |
| rust_score | SSIMULACRA2 score from Rust implementation (if available) |

These results can be analyzed to:
- Determine which format performs best at different quality targets
- Compare the accuracy of different SSIMULACRA2 implementations
- Develop optimized compression strategies for different image types
- Visualize quality-vs-size tradeoffs for each format

## How It Works

press_it uses SSIMULACRA2 to measure the perceptual quality of compressed images. When you provide a target quality score (0-100), the tool:

1. Converts your input image to a standardized format
2. Tests compression with different formats (JPEG, WebP, AVIF, PNG)
3. Performs a binary search to find the optimal quality parameter for each format
4. Selects the format that produces the smallest file size while meeting your target quality
5. Outputs the optimized image in the selected format

For detailed information about SSIMULACRA2 and how to interpret quality scores, please refer to the [original SSIMULACRA2 implementation](https://github.com/libjxl/libjxl/tree/main/tools/ssimulacra2) in the JPEG XL repository.

## Requirements

- Python 3.8+
- SSIMULACRA2 0.2.2+
- Pillow
- tqdm
- requests

## License

This project is licensed under the GNU General Public License v3 (GPL-3.0) - see the LICENSE file for details.
