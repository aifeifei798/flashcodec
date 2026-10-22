"""VRAM/storage benchmark: compares raw KV against each codec mode at 4K/16K/64K
sequence lengths, reporting actual cache occupancy (tensor storage bytes) and
fill/read timings.

Usage: .venv/bin/python benchmarks/bench_vram.py [--cuda]
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from flashcodec import CodecKVCache

DEVICE = "cuda" if (torch.cuda.is_available() and "--cuda" in sys.argv) else "cpu"
B, H, D = 1, 8, 64
MODES = [
    ("raw (fp32)", None),
    ("jpeg k=2 (4x)", dict(mode="jpeg", keep_freqs=2)),
    ("jpeg k=4 (2x)", dict(mode="jpeg", keep_freqs=4)),
    ("jpeg k=2 int8", dict(mode="jpeg", keep_freqs=2, quant_bits=8)),
    ("jpeg k=4 int8", dict(mode="jpeg", keep_freqs=4, quant_bits=8)),
    ("png keep=0.5", dict(mode="png", keep_ratio=0.5)),
    ("png keep=1.0", dict(mode="png", keep_ratio=1.0)),
]


def mb(n):
    return n / 1024 / 1024


print(f"device={DEVICE}  B={B} H={H} D={D}  block=8  fp32\n")
header = (f"{'seq':>6} | {'mode':<14} | {'storage':>10} | {'ratio':>6} | "
          f"{'fill':>8} | {'read':>8}")
print(header)
print("-" * len(header))


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def timed(fn):
    """Wall time of one call, properly synchronized: CUDA launches are async,
    so without events/synchronize the numbers are meaningless."""
    if DEVICE == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / 1e3
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


for S in (4096, 16384, 65536):
    torch.manual_seed(0)
    k = torch.randn(B, H, S, D, device=DEVICE)
    v = torch.randn(B, H, S, D, device=DEVICE)
    raw_bytes = (k.numel() + v.numel()) * k.element_size()
    for name, kw in MODES:
        if DEVICE == "cuda":
            base_alloc = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        if kw is None:
            fill_s = read_s = 0.0
            store, ratio = raw_bytes, 1.0
        else:
            cache = CodecKVCache(block_size=8, **kw)
            cache.forward(k, v)
            sync()                              # warmup (cuBLAS/kernel init)
            fill_s = min(timed(lambda: cache.forward(k, v))
                         for _ in range(3))
            st = cache.stats(0)
            store, ratio = st["storage_bytes"], st["storage_bytes"] / st["raw_bytes"]
            cache.key_cache[0].sum().item()
            sync()
            read_s = min(timed(lambda: cache.key_cache[0].sum().item())
                         for _ in range(3))
        peak = (torch.cuda.max_memory_allocated() / 1024 / 1024 - base_alloc / 1024 / 1024
                if DEVICE == "cuda" else float("nan"))
        print(f"{S:>6} | {name:<14} | {mb(store):>8.2f}MB | {ratio:>6.2f} | "
              f"{fill_s * 1e3:>6.1f}ms | {read_s * 1e3:>6.1f}ms"
              + (f"  (peak +{peak:.0f}MB)" if DEVICE == "cuda" else ""))
    print()
