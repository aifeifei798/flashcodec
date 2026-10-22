"""flashcodec codecs: jpeg (lossy frequency domain) and
png (lossless/near-lossless time domain)."""

from .jpeg import (JPEGCodec, dct_basis, dct_compress, dct_decompress,
                   dequantize_coeffs, quantize_coeffs)
from .png import PNGCodec, png_filter, png_reconstruct

__all__ = [
    "JPEGCodec", "PNGCodec",
    "dct_basis", "dct_compress", "dct_decompress",
    "quantize_coeffs", "dequantize_coeffs",
    "png_filter", "png_reconstruct",
]
