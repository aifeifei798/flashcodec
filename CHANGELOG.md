# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The import name is always `flashcodec`; the PyPI distribution is
`flashcodec-kv` (the bare `flashcodec` name on PyPI is an unrelated 0.0.0
placeholder package).

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
  `flashcodec` name on PyPI is taken by an unrelated placeholder package.
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
