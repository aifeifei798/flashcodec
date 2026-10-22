"""flashcodec — compress time-series KV cache tensors as "high-dimensional
multimedia image signals".

Two classic compression theories ported to Transformer KV cache management:
- mode="jpeg": blockwise 1D DCT-II over an orthonormal basis, lossy
  frequency-domain compression (low-frequency skeleton);
- mode="png": causal blockwise prediction deltas, lossless/near-lossless
  time-domain compression (anchor + Δ).

Tensor contract: KV layout (B, H, S, D), compression axis dim=-2 (seq axis).
"""

from .cache import HAS_HF_CACHE, CodecKVCache, FlashCodecCache, \
    FlashCodecLayer, make_codec
from .codecs import JPEGCodec, PNGCodec, dct_basis, dct_compress, \
    dct_decompress, dequantize_coeffs, png_filter, png_reconstruct, \
    quantize_coeffs
from .patch import apply_flash_codec, find_attention_modules, \
    remove_flash_codec, undo_flash_codec

__version__ = "0.3.0"

__all__ = [
    "apply_flash_codec", "remove_flash_codec", "undo_flash_codec",
    "find_attention_modules",
    "CodecKVCache", "FlashCodecCache", "FlashCodecLayer", "make_codec",
    "HAS_HF_CACHE", "JPEGCodec", "PNGCodec",
    "dct_basis", "dct_compress", "dct_decompress",
    "quantize_coeffs", "dequantize_coeffs",
    "png_filter", "png_reconstruct",
    "__version__",
]
