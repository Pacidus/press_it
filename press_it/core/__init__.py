"""Core functionality for press_it."""

from press_it.core.encoders import (
    get_encoder,
    get_extension,
    encode_mozjpeg,
    encode_webp,
    encode_avif,
    encode_png,
)

from press_it.core.quality import quality_optimizer, optimize_all_formats

from press_it.core.compression import (
    initialize_compression,
    compress_with_target_quality,
    bulk_compress,
)

# Define what's available when doing "from press_it.core import *"
__all__ = [
    # Encoders
    "get_encoder",
    "get_extension",
    "encode_mozjpeg",
    "encode_webp",
    "encode_avif",
    "encode_png",
    # Quality optimization
    "quality_optimizer",
    "optimize_all_formats",
    # Compression workflows
    "initialize_compression",
    "compress_with_target_quality",
    "bulk_compress",
]
