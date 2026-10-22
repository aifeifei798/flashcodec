# flashcodec

> Compressing Transformer KV Cache tensors as **high-dimensional multimedia image signals**.

[English](README.md) | [简体中文](README.zh-CN.md)

Conventional LLM research treats hidden-space tensors as isolated scalar
factors. flashcodec observes that the seq axis of a `(B, H, S, D)` KV tensor
behaves like the rows/columns of a high-dimensional image — adjacent token
representations are strongly autocorrelated, energy concentrates in low
frequencies, and first-order differences collapse in variance. Two classic
multimedia compression theories therefore transfer directly:

| Mode | Inspired by | Transform | Nature | Storage |
|---|---|---|---|---|
| `mode="jpeg"` | JPEG | Block-wise orthonormal 1D DCT-II, keep lowest K coeffs | **Lossy, frequency domain** | `K/r` (4×–8× less memory) |
| `mode="png"` | PNG | Per-block anchor (I-Token) + first-order deltas, Top-M pruning | **Lossless / near-lossless, time domain** | Dense, size-neutral; sparsity left to entropy coding |

Math (block size `r`, operating axis `dim=-2`):

- **JPEG**: $X_{\text{blk}} \in \mathbb{R}^{r \times D}$, $C = M_{\text{enc}} X_{\text{blk}}$ with
  $M_{\text{enc}} \in \mathbb{R}^{K \times r}$, $M[j,n] = c_j \cos\frac{\pi}{r}(n+\tfrac12)j$,
  orthonormal rows ($K=r$ is per-block lossless); reconstruct $\hat X = M_{\text{enc}}^\top C$.
- **PNG**: $\Delta_i = x_i - x_{i-1}$ ($i \ge 1$), $x_0$ kept at full precision as the
  anchor; rebuild $\hat x_i = x_0 + \sum_{j=1}^{i} \Delta_j$ (`cumsum`, purely
  linear and autograd-differentiable). Autocorrelation implies
  $\mathrm{Var}(\Delta)/\mathrm{Var}(X) \approx 0.03$ (**30× variance collapse**,
  measured); after Top-M pruning the residual is nearly zero and sparse.

```
                    ┌────────────────────────────────────────────┐
  (B,H,S,D) ──append──▶  raw_buffer (tail < r, full fidelity)     │
                    │        │ r tokens accumulated, batched      │
                    │        ▼ archive (single GEMM)              │
                    │  compressed_storage                         │
                    │   jpeg: low-freq coeffs (B,H,n,K,D)         │
                    │   png:  (deltas (B,H,n,r-1,D), anchor)      │
                    └───────────────┬────────────────────────────┘
                                    │ read(): per-block decode + raw tail
                                    ▼
                        reconstructed KV view (B,H,S,D)
```

## Causal-safety design

Only **complete blocks** are archived; the current incomplete block always
stays in `raw_buffer` at original precision. During incremental decoding, no
future perturbation can ever rewrite archived tensors (verified bitwise in
tests). On the lossless tier (`mode="png", keep_ratio=1.0`), step-by-step
decoded attention matches full-sequence attention within ≤1e-4.

## Installation

```bash
pip install -e .        # dependency: torch>=2.0
```

## Quick start

```python
import torch
from flashcodec import FlashCodecCache, apply_flash_codec

# 1) Native: a first-class transformers 5.x Cache backend (any CausalLM)
cache = FlashCodecCache(mode="jpeg", block_size=8, keep_freqs=2)  # 4×
out = model.generate(input_ids, max_new_tokens=64,
                     past_key_values=cache, use_cache=True)
print(cache.get_seq_length(), cache.storage_bytes())

# 2) One-line hot patch: every generate() defaults to the codec cache
apply_flash_codec(model, mode="png", block_size=8, keep_ratio=1.0)
out = model.generate(input_ids, max_new_tokens=64)
# PNG lossless tier reproduces the unpatched model's tokens EXACTLY
from flashcodec import remove_flash_codec
remove_flash_codec(model)          # restore default behavior

# 3) Standalone (no transformers needed): direct cache manager
from flashcodec import CodecKVCache
cache = CodecKVCache(mode="jpeg", block_size=8, keep_freqs=2)
k, v = cache.forward(k_seq, v_seq)          # full-sequence fill
k, v = cache.decode_step(new_k, new_v)      # incremental decode
```

`FlashCodecCache` subclasses `transformers.cache_utils.Cache` (5.x layer
API): write/read replacement is converged at the `cache.update(k, v,
layer_idx)` interface — zero model surgery, works with any CausalLM
(Llama/Qwen/Mistral-style GQA included).

## Benchmarks (measured; B=2 H=8 S=2048 D=64, random-walk autocorrelated signal, fp32)

### Reconstruction quality (`benchmarks/bench_reconstruct.py`)

| Config | PSNR (dB) | cosine | Storage |
|---|---|---|---|
| jpeg r=8 k=1 (8×) | 42.95 | 0.99943 | 12.5% |
| jpeg r=8 k=2 (4×) | 47.22 | 0.99985 | 25.0% |
| jpeg r=8 k=4 (2×) | 51.66 | 1.00000 | 50.0% |
| png r=8 keep=0.5 | 50.11 | 0.99997 | 100% (sparsity → entropy coder) |
| png r=8 keep=1.0 | ~196 (fp rounding) | 1.00000 | 100% |

### Cache footprint & latency (`benchmarks/bench_vram.py`, CPU)

| seq | Mode | storage | ratio | fill | read |
|---|---|---|---|---|---|
| 4096 | jpeg k=2 | 4.0MB | 0.25 | 8.3ms | 2.4ms |
| 16384 | jpeg k=2 | 16.0MB | 0.25 | 28.3ms | 13.0ms |
| 65536 | jpeg k=2 | 64.0MB | 0.25 | 132.6ms | 50.2ms |
| 65536 | png keep=1.0 | 256.0MB | 1.00 | 196.3ms | 82.9ms |

PNG does not shrink dense storage (the fate of a lossless tier) — its value is
the 30× variance collapse plus near-zero sparse residuals after Top-M, i.e.
ideal input for an entropy coder (none is bundled).

## Repository layout

```
flashcodec/
├── flashcodec/
│   ├── __init__.py        # apply_flash_codec / CodecKVCache / JPEGCodec / PNGCodec
│   ├── codecs/jpeg.py     # DCT-II basis, project/reconstruct (pure GEMM, no loops)
│   ├── codecs/png.py      # anchor + first-order delta, cumsum, Top-M filter
│   ├── cache.py           # CodecKVCache: raw tail + block archives, full/incr interfaces
│   └── patch.py           # one-line hot patch: cache-level injection (HF Cache duck-typing)
├── tests/                 # jpeg / png / causal suites (standalone scripts or pytest)
├── benchmarks/            # storage benchmark + PSNR/cosine quality benchmark
├── pyproject.toml
└── README.md
```

## Tests

```bash
python tests/test_jpeg.py           # orthonormality, k=r lossless, k=2 cosine>0.99, K monotone
python tests/test_png.py            # lossless maxdiff<1e-6, variance ratio printed, differentiable
python tests/test_causal.py         # zero future leakage, incremental vs full ≤1e-4
python tests/test_hf_integration.py # native transformers 5.x generate: explicit cache / hot patch /
                                    # PNG lossless exact tokens / 70-token multi-block stability
```

## Boundaries (honest list)

- Research prototype: `read()` materializes the full reconstructed view for
  attention; memory gains are at the **cache storage layer** (table above);
  end-to-end benefits require combining with materialization-free kernels
  (flash-attention style).
- Full-sequence `forward(seq)` prefill compresses complete blocks directly, so
  mid-block attention sees the block-level reconstruction (within-block
  approximation); the incremental `decode_step` path has no such issue.
- Compressed tensors are dense layouts; PNG Top-M sparsity is not yet wired to
  an entropy coder.
- Same lineage as the v4.2 prototypes inside
  [Mini-Architecture](https://github.com/aifeifei798/Mini-Architecture)
  (there: per-position view semantics for cross-layer KV reuse; here:
  standard cache storage semantics).

## License

[MIT](LICENSE)
