# press_it

A tool to compress images with formats targeting a specific perceptual quality.

## Features

- Automatically selects the best compression format (JPEG, WebP, AVIF, PNG) for your target quality
- Uses SSIMULACRA2 for perceptual quality measurement
- Employs binary search to find the optimal compression parameters
- Command-line interface for easy integration into scripts and workflows

## Installation

### Prerequisites

The following tools need to be installed on your system:

- ImageMagick (`magick`)
- MozJPEG (`cjpeg`)
- WebP tools (`cwebp`, `dwebp`)
- AVIF tools (`avifenc`, `avifdec`)
- PNG optimization tools (`pngcrush`)

After installing the prerequisites, you can install press_it:

```bash
pip install .
```

Or from PyPI:

```bash
pip install press-it
```

## Usage

```bash
press-it input.png 85.0
```

This will create a compressed version of `input.png` with a perceptual quality of about 85.0 SSIM. The tool automatically selects the format that gives the smallest file size while meeting the quality target.

### Options

```
usage: press-it [-h] [--version] [--resize RESIZE] [-o OUTPUT] input_image target_ssim

Optimize images for target SSIM using multiple encoders

positional arguments:
  input_image           Path to source image file
  target_ssim           Target SSIM value (0-100)

optional arguments:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
  --resize RESIZE, -r RESIZE
                        Resize the image (width)x(height). If one dimension is not set, the aspect ratio is respected
  -o OUTPUT, --output OUTPUT
                        Output directory. If not provided, uses current directory
```

## How It Works

1. The input image is converted to PNG format for consistent quality measurement
2. The tool tests compression with different formats (JPEG, WebP, AVIF, PNG)
3. For each format, it performs a binary search to find the minimal quality setting that meets the target SSIM score
4. The format with the smallest file size that meets the quality target is selected
5. The optimized image is saved to the output directory

## License

This project is licensed under the GNU General Public License v3 (GPL-3.0) - see the LICENSE file for details.
