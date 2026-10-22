"""Reconstruction-quality benchmark: PSNR and cosine similarity per mode.

Signal: synthetic autocorrelated latents (random walk, mimicking the strong
temporal correlation of real KV caches).
Usage: .venv/bin/python benchmarks/bench_reconstruct.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from flashcodec import CodecKVCache

torch.manual_seed(0)
B, H, S, D = 2, 8, 2048, 64
x = torch.cumsum(torch.randn(B, H, S, D) * 0.1, dim=-2)   # autocorrelated latents


def psnr(ref, hat):
    rmse = ((ref - hat) ** 2).mean().sqrt().item()
    peak = ref.abs().max().item()
    return 20 * math.log10(peak / max(rmse, 1e-12))


def cos(ref, hat):
    return torch.nn.functional.cosine_similarity(
        ref.flatten(), hat.flatten(), dim=0).item()


CONFIGS = [
    ("jpeg", dict(mode="jpeg", keep_freqs=1)),
    ("jpeg", dict(mode="jpeg", keep_freqs=2)),
    ("jpeg", dict(mode="jpeg", keep_freqs=4)),
    ("jpeg", dict(mode="jpeg", keep_freqs=2, quant_bits=8)),
    ("jpeg", dict(mode="jpeg", keep_freqs=4, quant_bits=8)),
    ("png", dict(mode="png", keep_ratio=0.25)),
    ("png", dict(mode="png", keep_ratio=0.5)),
    ("png", dict(mode="png", keep_ratio=1.0)),
]

print(f"signal: random-walk B={B} H={H} S={S} D={D} (autocorrelated latent simulation)\n")
header = f"{'config':<16} | {'PSNR(dB)':>9} | {'cosine':>8} | {'storage':>8}"
print(header)
print("-" * len(header))
st = cache = None
for _, kw in CONFIGS:
    cache = CodecKVCache(block_size=8, **kw)
    hat, _ = cache.forward(x, x)
    s = cache.stats(0)
    name = (f"{kw['mode']} r=8"
            + (f" k={kw['keep_freqs']}" if kw["mode"] == "jpeg"
               else f" keep={kw['keep_ratio']}")
            + (" int8" if kw.get("quant_bits") else ""))
    print(f"{name:<16} | {psnr(x, hat):>9.2f} | {cos(x, hat):>8.5f} | "
          f"{s['storage_bytes'] / s['raw_bytes']:>7.2%}")
