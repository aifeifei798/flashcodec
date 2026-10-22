"""PNG-style lossless/near-lossless time-domain compression: causal blockwise
prediction delta filtering.

Physical intuition: the hidden vectors of adjacent tokens are highly
autocorrelated; after a first-order delta Δ = x_i - x_{i-1} the variance drops
sharply and the values cluster near 0 (very low information entropy). Each
block stores a full-precision anchor at its head (I-Token) and deltas
elsewhere; cumsum gives causal reconstruction that is bitwise invertible at
keep_ratio=1.0 (up to fp rounding).
When keep_ratio<1.0, the Top-M steps by |Δ| magnitude are kept per
(block, feature), emulating a near-lossless mode.

Tensor contract: input (..., S, D); the operating axis is the second-to-last
dimension (seq axis, dim=-2). Purely linear operators + discrete selection, so
autograd works naturally (gradients flow only to the retained delta steps).
"""

import math

import torch
import torch.nn.functional as F


def _pad_to_blocks(x, r):
    """Zero-pad the seq axis (-2) to a multiple of r; returns (block tensor,
    original length)."""
    S = x.shape[-2]
    n = (S + r - 1) // r
    pad = n * r - S
    if pad:
        x = F.pad(x, (0, 0, 0, pad))
    blocks = x.reshape(*x.shape[:-2], n, r, x.shape[-1])
    return blocks, S


def png_filter(x, block_size, keep_ratio=1.0):
    """Delta filtering: x (..., S, D) -> (deltas (..., n, r-1, float32),
    anchor (..., n, D, input dtype)).
    Position 0 of every block is forced to a full-precision anchor; position
    i>=1 stores Δ_i = x_i - x_{i-1}.
    When keep_ratio<1.0, the Top-M steps by |Δ| magnitude are kept per
    (block, feature) (m = ceil((r-1)*keep_ratio), m>=1), the rest are set to 0.
    dtype contract: **deltas are always fp32** — computing the deltas of bf16
    inputs in bf16 loses ~0.4% precision and breaks the bitwise exactness of
    lossless reconstruction; Δ is a "portable representation" whose precision
    must not be lower than that of the original signal (bf16 deltas are exactly
    representable in fp32, so cumsum restores them bitwise).
    The anchor keeps the input dtype."""
    assert 0 < keep_ratio <= 1.0, "keep_ratio must be within (0, 1]"
    blocks, _ = _pad_to_blocks(x, block_size)
    anchor = blocks[..., 0, :]                                # (..., n, D)
    bf = blocks.float()
    delta = bf[..., 1:, :] - bf[..., :-1, :]                  # (..., n, r-1, D)
    if keep_ratio < 1.0:
        m = max(1, math.ceil(delta.shape[-2] * keep_ratio))
        topi = delta.abs().topk(m, dim=-2).indices
        mask = torch.zeros_like(delta, dtype=torch.bool)
        mask.scatter_(-2, topi, True)
        delta = delta * mask
    return delta, anchor


def png_reconstruct(deltas, anchor):
    """Causal reconstruction: anchor + cumsum (accumulated in fp32, then cast
    back to the anchor dtype).
    Position 0 = anchor, position i>=1 = anchor + the first i delta steps.
    deltas (..., n, r-1, D), anchor (..., n, D) -> x (..., n*r, D).
    With bf16 input + keep_ratio=1.0 the result is bitwise lossless (deltas are
    exact in fp32)."""
    base = anchor.float().unsqueeze(-2)                       # (..., n, 1, D)
    rec = torch.cat([base, base + torch.cumsum(deltas.float(), dim=-2)], dim=-2)
    return rec.to(anchor.dtype).reshape(*rec.shape[:-3],
                                        rec.shape[-3] * rec.shape[-2],
                                        rec.shape[-1])


class PNGCodec:
    """PNG-style prediction-filter codec (stateless, shareable across caches)."""

    name = "png"

    def __init__(self, block_size=8, keep_ratio=1.0):
        assert 0 < keep_ratio <= 1.0, "keep_ratio must be within (0, 1]"
        self.block_size = block_size
        self.keep_ratio = keep_ratio

    def filter(self, x):
        return png_filter(x, self.block_size, self.keep_ratio)

    def reconstruct(self, deltas, anchor):
        return png_reconstruct(deltas, anchor)

    def storage_ratio(self):
        """Dense tensor storage ratio: 1.0 for fp32 input (anchor + (r-1) delta
        steps are the same size).
        With bf16 input, deltas are stored as fp32 (the dtype contract for
        bitwise losslessness), so the archive is about 1.9× — bandwidth traded
        for exactness, with the compression headroom left to an entropy coder.
        PNG's value lies at the information-entropy level: delta variance drops
        ~30×, and after Top-M truncation the residual is highly sparse and
        near zero — that is the headroom for an entropy coder
        (run-length/arithmetic coding)."""
        return 1.0

    def __repr__(self):
        return (f"PNGCodec(block_size={self.block_size}, "
                f"keep_ratio={self.keep_ratio})")
