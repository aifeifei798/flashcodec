# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The import name is always `flashcodec`; the PyPI distribution is
`flashcodec-kv` (the bare `flashcodec` name on PyPI is an unrelated 0.0.0
placeholder package).

## [0.5.0] — 2026-10-23

### Added

- **HF integration for materialization-free decode**:
  `apply_flash_codec(..., materialization_free=True)` wraps standard
  Llama/Qwen-style attention forwards so decode steps attend on the stored rows
  (projection + RoPE unchanged, guarded to batch-1 / single-token / eval /
  non-sliding layers, silent fallback otherwise). The prefill view is dropped on
  the first decode step. `remove_flash_codec` restores everything;
  `codec_attention_stats(model)` reports the wrapped module count.
- `benchmarks/bench_e2e.py`: long-context end-to-end benchmark (speed, peak and
  steady memory). Qwen3-0.6B, 4096-token prompt + 32 new tokens:
  baseline 53.2 tok/s / peak 581MB; codec k=2 int8 materialized 44.9 tok/s /
  steady 1535MB; materialization-free 30.7 tok/s / peak 642MB / steady 122MB
  (12.6× less steady memory, ~1.7× slower decode).
- `tests/test_hf_integration.py` covers the patched generate path (lossless tier
  still token-identical, no reconstruction view retained, clean removal).

## [0.4.0] — 2026-10-23

### Added

- **`flashcodec/attention.py` — materialization-free decode attention.** Both
  codecs are linear along the seq axis (`K_block = A @ S`), so a decode query
  can attend on the stored rows: `q·K[p] = (q·S^T) @ A[p]` and
  `out = (prob @ A) @ S`. Measured at 64K with jpeg k=2 int8: 0.86ms / 520MB
  (materialized) → 0.46ms / 128MB (coefficient domain), maxdiff ≤ 1e-6.
  Decode only; png is numerically supported but has no memory win.
- **`per_layer` budgets** on `FlashCodecCache.from_config`, plus
  `benchmarks/bench_adaptive.py` (per-layer sensitivity calibration).
  Measured on Qwen3-0.6B: adaptive 40.83 ppl at 75% storage vs uniform k=6
  48.05 at 75% and k=r int8 19.36 at 50% — calibration helps over uniform
  truncation, but int8 remains the default because the damages are
  super-additive.
- `tests/test_attention.py` and a `benchmarks/bench_attention.py` memory/latency
  benchmark.

## [0.3.0] — 2026-10-23

### Added

- **`quant_bits=8` for the jpeg codec**: coefficients are stored as int8 with a
  per-(block, frequency) fp32 scale. A real byte-level saving on top of the
  `k/r` geometric ratio: 6.6% of fp32 raw KV at k=2, ~52% of bf16 raw KV at
  k=8 (full lossless DCT basis) with +2% perplexity on a dense model.
- **`benchmarks/bench_ppl.py`**: perplexity evaluation on a local corpus, run
  against a dense model (Qwen3-0.6B) and a hybrid one (Qwen3.5-0.8B).
- **Honest measurement fixes**: `bench_vram --cuda` now uses CUDA events plus a
  warmup (the old numbers were async and meaningless), and `run_qwen_demo`
  reports tokens/s and peak memory.
- README (EN/ZH): real-KV statistics and perplexity tables.

### Changed

- README claims about autocorrelation and "30× variance compression" are now
  qualified with measured real-model statistics; the dense-model caveat
  (coefficient truncation compounds per layer) is documented, with
  `k = r + quant_bits=8` as the safe tier for fully compressible models.

## [0.2.0] — 2026-10-23

### Added

- **Beam search support**: `FlashCodecLayer` / `CodecKVCache` implement
  `reorder_cache`, `batch_repeat_interleave` and `batch_select_indices` over
  the archives *and* the raw tail. On the lossless tier (`png keep_ratio=1.0`)
  `generate(num_beams>1)` is bitwise-identical to the unpatched model.
- **Exact `crop`** for speculative decoding: archived blocks are kept verbatim
  and only the boundary partial block returns to the raw tail, so repeated
  crops no longer re-compress the history (the old rebuild-from-view path added
  loss on every crop).
- **Memoized `read()`**: decoded archive blocks land in an append-only
  reconstruction buffer, so a decode step decodes one new block instead of the
  whole history (`bench_vram` S=64K, jpeg k=2: 38.7ms → 5.2ms).

### Changed

- Distribution renamed to `flashcodec-kv` (import name unchanged) because the
  `flashcodec` name on PyPI is taken by an unrelated placeholder package. The
  GitHub repository moved to
  [aifeifei798/flashcodec-kv](https://github.com/aifeifei798/flashcodec-kv)
  (the old URL redirects).
- Four new regression tests: memoized read vs one-shot decode, exact and
  repeatable crop, beam-search batch gather, and beam search / crop parity
  against the unpatched model.

### Fixed

- README numbers that had gone stale: the Qwen3.5-0.8B table now matches a
  fresh `benchmarks/run_qwen_demo.py` run (jpeg k=2 → 32.3% storage / 38.1%
  token match, jpeg k=4 → 54.8% / 49.2%, png lossless → 1.8× with bitwise
  identical tokens), and the CPU latency table reflects the memoized read.

## [0.1.0] — 2026-10-23

### Added

- Initial release: JPEG-style blockwise DCT-II codec (`mode="jpeg"`, keep the
  lowest K coefficients) and PNG-style anchor + first-order-delta codec
  (`mode="png"`, lossless at `keep_ratio=1.0`).
- `CodecKVCache` (standalone, duck-typed as a cache), `FlashCodecCache` /
  `FlashCodecLayer` (first-class `transformers` 5.x Cache backend, hybrid
  architectures handled through `from_config`), and `apply_flash_codec` /
  `remove_flash_codec` one-line hot patch.
- Causal-safety guarantees (complete blocks only; archived tensors are never
  rewritten), capacity-doubling preallocated archive storage, benchmarks
  (`bench_reconstruct`, `bench_vram`, `run_qwen_demo`) and the Gradio
  side-by-side demo.
