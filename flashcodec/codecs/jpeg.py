"""JPEG-style lossy frequency-domain compression: blockwise orthonormal 1D
DCT-II.

Physical intuition: the hidden vectors of adjacent tokens form a strongly
autocorrelated time-series signal whose energy concentrates in low
frequencies. Each block is projected onto the first K orthonormal DCT-II basis
vectors (DC mean + AC linear trend, ...), filtering out high-frequency random
noise and yielding a geometric r/K compression ratio. Basis rows are
orthonormal, so keep=r is lossless per block.

Tensor contract: input (..., S, D); the compression/filtering axis is the
second-to-last dimension (seq axis, dim=-2). Everything is batched matmul
(GEMM), with no explicit Python loops.
"""

import math

import torch
import torch.nn.functional as F

_BASIS_CACHE = {}


def dct_basis(r, k, device=None, dtype=None):
    """Precompute the orthonormal DCT-II basis M_enc (k, r):
    M[j, n] = c_j · cos(π/r · (n+½)·j), c_0 = sqrt(1/r), c_j = sqrt(2/r).
    Rows are orthonormal (M @ M.T = I); k == r gives the complete orthonormal
    basis (lossless reconstruction).
    Module-level constant cache, keyed by (r, k)."""
    key = (r, k)
    if key not in _BASIS_CACHE:
        n = torch.arange(r, dtype=torch.float32)
        j = torch.arange(k, dtype=torch.float32)
        c = torch.where(j == 0, math.sqrt(1.0 / r), math.sqrt(2.0 / r))
        m = c[:, None] * torch.cos(math.pi / r * (n[None, :] + 0.5) * j[:, None])
        _BASIS_CACHE[key] = m
    m = _BASIS_CACHE[key]
    if device is not None or dtype is not None:
        m = m.to(device=device, dtype=dtype or m.dtype)
    return m


def _pad_to_blocks(x, r):
    """Zero-pad the seq axis (-2) to a multiple of r; returns (block tensor,
    original length)."""
    S = x.shape[-2]
    n = (S + r - 1) // r
    pad = n * r - S
    if pad:
        x = F.pad(x, (0, 0, 0, pad))          # (..., S, D) pads the -2 axis only
    blocks = x.reshape(*x.shape[:-2], n, r, x.shape[-1])
    return blocks, S


def dct_compress(x, r, k):
    """Blockwise DCT projection: x (..., S, D) -> coeffs (..., n_blocks, k, D)
    (x dtype).
    The seq axis is split into blocks of r (zero-padded at the tail); each
    block is projected along the within-block axis onto the first k DCT-II
    basis vectors (computed in fp32, then cast back to the input dtype — lossy
    modes need no higher precision, and the storage ratio stays honest).
    Compression storage ratio = k/r (same dtype, plus tail padding overhead)."""
    blocks, _ = _pad_to_blocks(x, r)
    basis = dct_basis(r, k, device=x.device, dtype=torch.float32)
    hp = blocks.float().transpose(-2, -1)              # (..., n, D, r)
    coef = torch.matmul(hp, basis.t())                 # (..., n, D, k)
    return coef.transpose(-2, -1).to(x.dtype)          # (..., n, k, D)


def dct_decompress(coeffs, r, out_len=None):
    """Blockwise DCT reconstruction: coeffs (..., n_blocks, k, D) ->
    x (..., n_blocks*r, D).
    Low-frequency coefficients are multiplied by the basis to reconstruct the
    within-block waveform; when out_len is given, the tail padding is trimmed."""
    n, k = coeffs.shape[-3], coeffs.shape[-2]
    basis = dct_basis(r, k, device=coeffs.device, dtype=torch.float32)
    hp = torch.matmul(coeffs.transpose(-2, -1).float(), basis)   # (...,n,D,r)
    rec = hp.transpose(-2, -1).reshape(
        *coeffs.shape[:-3], n * r, coeffs.shape[-1])
    if out_len is not None:
        rec = rec[..., :out_len, :]
    return rec


class JPEGCodec:
    """JPEG-style lossy frequency-domain codec (stateless, shareable across
    caches)."""

    name = "jpeg"

    def __init__(self, block_size=8, keep_freqs=2):
        assert 1 <= keep_freqs <= block_size, \
            f"keep_freqs must be within [1, {block_size}]"
        self.block_size = block_size
        self.keep_freqs = keep_freqs

    def compress(self, x):
        return dct_compress(x, self.block_size, self.keep_freqs)

    def decompress(self, coeffs, out_len=None):
        return dct_decompress(coeffs, self.block_size, out_len)

    def storage_ratio(self):
        """Archive tensor element count / original element count (excluding the
        raw tail)."""
        return self.keep_freqs / self.block_size

    def __repr__(self):
        return (f"JPEGCodec(block_size={self.block_size}, "
                f"keep_freqs={self.keep_freqs})")
