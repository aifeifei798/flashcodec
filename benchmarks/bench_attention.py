"""Decode-attention benchmark: materialized KV view vs coefficient-domain
attention (no KV reconstruction).

For one layer and one decode query, measures per-step latency, peak extra
memory and steady-state resident memory. Each path gets freshly built streams so
the numbers are not contaminated by the other path's buffers.

Usage: python benchmarks/bench_attention.py [--cuda]
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from flashcodec.attention import codec_decode_attention
from flashcodec.cache import _CodecStream, make_codec

DEVICE = "cuda" if (torch.cuda.is_available() and "--cuda" in sys.argv) else "cpu"
B, H, D = 1, 8, 64
G = 2                                    # GQA: 16 query heads, 8 kv heads
BLOCK = 8
MODES = [
    ("jpeg k=2 int8", dict(mode="jpeg", block_size=BLOCK, keep_freqs=2,
                           quant_bits=8)),
    ("png keep=1.0", dict(mode="png", block_size=BLOCK, keep_ratio=1.0)),
]


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def build(cfg, k, v):
    ks = _CodecStream(make_codec(**cfg))
    vs = _CodecStream(make_codec(**cfg))
    ks.append(k)                          # append only, no read(): _rec stays None
    vs.append(v)
    return ks, vs


def timed(fn, repeat=10):
    best = float("inf")
    for _ in range(3):
        if DEVICE == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repeat):
                fn()
            end.record()
            torch.cuda.synchronize()
            best = min(best, start.elapsed_time(end) / repeat)
        else:
            t0 = time.perf_counter()
            for _ in range(repeat):
                fn()
            best = min(best, (time.perf_counter() - t0) / repeat * 1e3)
    return best


def mem_mb(fn):
    """(peak extra MB during fn, resident extra MB after fn)."""
    if DEVICE != "cuda":
        return float("nan"), float("nan")
    sync()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    sync()
    peak = (torch.cuda.max_memory_allocated() - base) / 1024 / 1024
    resident = (torch.cuda.memory_allocated() - base) / 1024 / 1024
    return peak, resident


print(f"device={DEVICE}  B={B} H_kv={H} H_q={H * G} D={D}  block={BLOCK}  fp32\n")
header = (f"{'seq':>6} | {'mode':<14} | {'path':<12} | {'latency':>9} | "
          f"{'peak':>8} | {'resident':>9} | {'maxdiff':>9}")
print(header)
print("-" * len(header))

for S in (4096, 16384, 65536):
    torch.manual_seed(0)
    k = torch.randn(B, H, S, D, device=DEVICE)
    v = torch.randn(B, H, S, D, device=DEVICE)
    q = torch.randn(B, H * G, 1, D, device=DEVICE)
    scale = D ** -0.5

    for name, cfg in MODES:
        def materialized(ks, vs):
            def fn():
                kk = ks.read().repeat_interleave(G, dim=1)
                vv = vs.read().repeat_interleave(G, dim=1)
                p = torch.softmax((q @ kk.transpose(-2, -1)) * scale, dim=-1)
                return p @ vv
            return fn

        ks, vs = build(cfg, k, v)
        ref = materialized(ks, vs)()
        t_mat = timed(materialized(ks, vs))
        p_mat, r_mat = mem_mb(materialized(ks, vs))
        print(f"{S:>6} | {name:<14} | {'materialize':<18} | {t_mat:>7.2f}ms | "
              f"{p_mat:>6.0f}MB | {r_mat:>7.0f}MB | {'':>9}")
        del ks, vs
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        impls = ["torch"] + (["triton"] if DEVICE == "cuda" else [])
        for impl in impls:
            ks, vs = build(cfg, k, v)
            out = codec_decode_attention(q, ks, vs, impl=impl)
            d = (out - ref).abs().max().item()
            t_coef = timed(lambda: codec_decode_attention(q, ks, vs, impl=impl))
            p_coef, r_coef = mem_mb(
                lambda: codec_decode_attention(q, ks, vs, impl=impl))
            assert ks._rec is None, \
                "the coefficient path must not materialize the view"
            print(f"{S:>6} | {name:<14} | {'coefficient-' + impl:<18} | "
                  f"{t_coef:>7.2f}ms | {p_coef:>6.0f}MB | {r_coef:>7.0f}MB | "
                  f"{d:>9.1e}")
            del ks, vs
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
    print()

print("note: 'materialize' allocates the (S, D) reconstructed view (persisted by\n"
      "read()'s memoization); the coefficient path decodes nothing, so its memory\n"
      "is the archive + scores. maxdiff is vs the materialized result (same\n"
      "reconstruction, different operation order).")
