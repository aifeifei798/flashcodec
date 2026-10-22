# AGENTS.md — flashcodec

Standalone open-source project: compresses Transformer time-series KV cache tensors
as "high-dimensional multimedia image signals" (jpeg = frequency-domain DCT lossy /
png = time-domain delta lossless). Pure torch, no pytest dependency, no CI.
Read this whole file before making changes.

## Running conventions

- **You must run from inside this directory (`flashcodec/`)**: the package and the
  project directory share a name, so `import flashcodec` from the parent directory
  only hits the namespace package (no `__version__`/exports) — an inherent trap of
  this same-name layout. After `pip install -e .` it imports from anywhere.
- Tests are plain runnable scripts (`assert` + `__main__` summary, function names
  are pytest-compatible): `python tests/test_jpeg.py`, `python tests/test_png.py`,
  `python tests/test_causal.py`, `python tests/test_hf_integration.py`.
  Run all four after touching codecs/cache.
- Benchmarks: `python benchmarks/bench_reconstruct.py` (PSNR/cosine),
  `python benchmarks/bench_vram.py [--cuda]` (storage occupancy/timings).
- This directory is gitignored inside the host repo — it is an independent git
  repository with independent releases.

## transformers integration (only targets local 5.20; 4.x explicitly unsupported)

- `FlashCodecLayer(DynamicLayer)` + `FlashCodecCache(Cache,
  layer_class_to_replicate=partial(...))`: 5.x's Cache is a layer container, and
  the base class lazily creates layers and hands them down to the layer; **5.x
  attention consumes the return value of `update` directly** (it does not read
  `key_cache`), so layer.update must return the reconstructed full causally-visible
  KV.
- `apply_flash_codec`'s main hook is `model.generate` (5.x generate builds its own
  DynamicCache internally, so injection only takes effect at the entry point); the
  forward wrapper is secondary.
- Generation semantics: **cache length = prompt + new_tokens - 1** — the last
  generated token's KV is never computed (nobody attends to it), so don't write the
  assertion as the total length.
- Integration-test traps: a randomly initialized tiny model will emit eos and stop
  early (disable it with `generation_config.eos_token_id = None`); scaling lm_head
  by 20 creates a clear argmax margin, isolating tie jitter and focusing on the
  cache mechanism.
- `crop` (speculative decoding path): rebuilds the archive from the cropped view
  (prototype semantics); `offload/prefetch` are no-ops; beam search
  (reorder_cache) is unsupported.

## dtype contract (distilled from Qwen3.5-0.8B practice; read before changing codecs)

- **PNG deltas are always fp32**: computing the deltas of bf16 inputs in bf16 loses
  ~0.4% precision, and cumsum reconstruction is no longer bitwise-exact (bf16
  deltas are exactly representable in fp32, so anchor+ΣΔ restores the original
  values bitwise) — this is the basis for "lossless → generated tokens are
  bitwise-identical to the baseline". Under bf16 deployment the archive is about
  1.9× dense (bandwidth traded for exactness).
- **JPEG coeffs are stored in the input dtype** (computed in fp32, then cast):
  lossy modes need no higher precision, and the storage ratio isn't inflated
  (k=2 measured ~27% raw).
- **`_CodecStream.read()` uniformly restores the stream's input dtype**: SDPA
  requires q/k/v to share a dtype, so feeding fp32 views to a bf16 model crashes
  outright.
- Hybrid architecture (Qwen3.5: 18 linear_attention + 6 full_attention): you must
  use `FlashCodecCache.from_config(model.config)` to attach a codec only to
  full_attention layers per layer_types, leaving the other layers as native dynamic
  layers (recursive state and other interfaces are unaffected);
  `apply_flash_codec` already routes through from_config automatically.
- Practical script: `benchmarks/run_qwen_demo.py` (PNG lossless bitwise-identity
  assertion + JPEG storage-ratio/fluency report + hybrid-layer audit).

## Tensor contract (read before changing any signature)

- KV layout `(B, H, S, D)`; **the compression/filtering axis is always dim=-2 (the
  seq axis)**.
- Codecs may only use batched matmul/elementwise operators; **explicit Python for
  loops are forbidden**; the trailing partial block is zero-padded
  (`_pad_to_blocks`) and trimmed back via `out_len`.
- Codecs compute in fp32 uniformly (`dct_compress` and friends call `.float()`
  internally); DCT with keep=r and PNG with keep_ratio=1 are the "lossless"
  baselines (fp round-trip on the ~1e-6/1e-7 level; see the tests for assertion
  tolerances, and don't tighten them to bitwise).

## Causal-safety invariants (breaking these is a bug)

1. **Archive complete blocks only**; an incomplete block always stays in
   `raw_buffer` (original precision) — a future perturbation at any point during
   incremental decoding must never rewrite already-archived tensors (`test_causal`
   verifies this bitwise).
2. The archives and views produced by the two fill paths — full `forward(seq)` and
   stepwise `decode_step` — must be **bitwise-identical** (both paths write into the
   same capacity-doubling storage and share a single decode pass, so they are
   inherently symmetric; think this through before switching to a sharded
   structure).
3. In the `mode="png", keep_ratio=1.0` lossless mode, stepwise decoding attention
   must align with the full forward pass within ≤1e-4 (the tests' correctness
   anchor).

## Design decisions (don't "refactor these away" in passing)

- **The HF hot patch lives in the Cache layer, not the attention kernel**: KV
  writes/reads in standard HF models converge on the Cache interface
  (`update`/`key_cache`), so injecting a `CodecKVCache` duck type is enough — far
  more robust than per-model surgery (GQA/MQA/RoPE variants).
- **PNG dense storage honestly counts as 1.0**: anchor+deltas are the same size as
  the original block; what it saves is information entropy (variance compression
  ~30×), and that headroom is left to an entropy coder — `storage_ratio()` must not
  "round" this into (r-1)/r.
- A full `forward(seq)` prefill compresses complete blocks directly, so in-block
  position attention sees the block-level reconstruction (intra-block
  approximation); the incremental path has no such issue. This is documented
  semantics, not a bug.
- Archive data structure: **a capacity-doubling preallocated single tensor**
  (std::vector style, starting at 16 blocks, doubling when full) — jpeg uses
  `_coeffs (B,H,capacity,k,D)`; png uses `_deltas (B,H,capacity,r-1,D) fp32 +
  _anchors (B,H,capacity,D)`. Each full block only does an in-place slice write;
  `read()` single-decodes the valid region + cats the raw tail.
  **Do not** revert to "append a new archive tensor per full block + loop over
  archives in read" — with that fragmentation, per-token decoding measured 170×
  slower reads (31ms → 0.18ms at S=4096, launch overhead dominating).
  `storage_bytes()` only counts the logical valid region, not the free capacity
  reserved by doubling. `reset()` zeroes the counters but **retains allocated
  capacity** (reused for crop/refill VRAM).

## Release

- `images/` holds the README screenshots and is maintained by the user — do not
  regenerate, replace, or commit new images unless explicitly asked.
- `pyproject.toml` (setuptools, MIT, torch>=2.0); the version number is written in
  both `pyproject.toml` and `flashcodec/__init__.py::__version__`, so changing one
  requires syncing the other.
- Before publishing to PyPI, run all three test groups + both benchmarks (the
  measured tables in the README must be updated accordingly — every number in the
  README was actually measured; no placeholders or stale data).
