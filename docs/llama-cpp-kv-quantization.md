# Same-bytes comparison with llama.cpp KV quantization

Measured 2026-10-23 on **Qwen3-0.6B** (28 layers, 8 KV heads, head_dim 128, i.e.
112 KiB of KV per token in fp16), RTX 5090 D, llama.cpp `184f65360`
(CUDA + flash-attention enabled), wikitext-2-raw.

## TL;DR

llama.cpp's own `-ctk q8_0` reaches **~53% of fp16 KV bytes at +0.05% perplexity**.
Both usable flashcodec tiers lose against it at equal bytes:

| flashcodec tier | bytes | damage at equal bytes vs `q8_0` |
|---|---|---|
| truncation (`k < r`) | 25–75% | **2.7× more bytes for the same damage**; `k=2` is unusable |
| lossless basis + `quant_bits=8` | 50% | **8× more damage** (+0.38% vs +0.05%) |

So the seq-axis DCT codec is **not** a drop-in improvement over llama.cpp's
row-wise KV quantization, and stacking the two does not help either: applied on
top of `q8_0` it reclaims 0.8% of the bytes for 63% more damage (§3). What the
measurements *do* show is that K and V are not equally sensitive, which changes
how a KV budget should be spent.

## 1. llama.cpp KV types (the target to beat)

`llama-perplexity -c 512 --chunks 40 -fa on -ngl 99`, perplexity:

| K | V | ppl | Δ vs f16 | B/token | of fp16 |
|---|---|---|---|---|---|
| f16 | f16 | **19.6037** | — | 114,688 | 100% |
| q8_0 | q8_0 | 19.6138 | +0.05% | 60,928 | 53.1% |
| q8_0 | f16 | 19.6141 | +0.05% | 87,808 | 76.6% |
| f16 | q8_0 | 19.6080 | +0.02% | 87,808 | 76.6% |
| q8_0 | q4_0 | 19.6579 | +0.28% | 46,592 | 40.6% |
| f16 | q4_0 | 19.6606 | +0.29% | 73,472 | 64.0% |
| q4_0 | f16 | 66.0552 | +237% | 73,472 | 64.0% |
| q4_0 | q8_0 | 66.5416 | +239% | 46,592 | 40.6% |
| q4_0 | q4_0 | 65.8321 | +236% | 32,256 | 28.1% |

Throughput (`llama-bench -p 4096 -n 32 -fa 1`, 3 reps):

| KV | pp4096 t/s | tg32 t/s |
|---|---|---|
| f16 | 68,121 | 713.7 |
| q8_0 | 61,857 | 610.5 |
| q4_0 | 62,022 | 610.7 |

Two asymmetries worth remembering:

- **4-bit K is the only cliff**: K alone at `q4_0` gives 66.06 (+237%), while V
  alone at `q4_0` costs +0.29%. If a KV budget has to be squeezed,
  `K=q8_0 + V=q4_0` is the cheap end point (40.6% of fp16, +0.28% ppl).
- 8-bit is free on both sides, so the *only* open design space in llama.cpp is
  K below 4 bits — precisely where the codec is weakest.

## 2. Same-bytes A/B: seq-axis DCT vs row-wise quantization

Same model, `transformers` 5.x, bf16, 64 × 512-token wikitext windows.
`q8_0 emul` = symmetric absmax quantization per 32 contiguous head dims
(llama.cpp semantics, not bit-exact). `dct` = this project's blockwise DCT-II
along the sequence axis, K-only or K+V.

| tier | ppl | Δ | of fp16 |
|---|---|---|---|
| baseline | 27.9690 | — | 100% |
| q8_0 K+V | 28.2485 | +1.00% | 53.1% |
| q8_0 K only | 27.9823 | **+0.05%** | 76.6% |
| dct b32 k32 int8 K only | 28.0750 | +0.38% | 75.8% |
| dct b16 k16 int8 K only | 28.1194 | +0.54% | 75.8% |
| dct b8 k8 int8 K only | 28.3300 | +1.29% | 75.8% |
| dct b8 k8 int8 K+V | 28.3241 | +1.27% | 51.6% |
| dct b8 k=4 K only | 172.26 | +516% | 75.0% |
| dct b8 k=4 K+V | 649.6 | +2222% | 50.0% |
| dct b8 k=2 K only | 3142 | +11100% | 62.5% |
| dct b8 k=2 K+V | 21257 | +75900% | 28.1% |

Absolute perplexities in the two tables are not comparable (HF runs disjoint
windows in bf16, llama.cpp runs contiguous chunks in fp16); only the relative
damage is.

## 3. Stacking the codec on top of llama.cpp's quantization

If llama.cpp already stores quantized KV, the natural follow-up is whether the
codec can compress *those* values further. Same harness (64 × 512 wikitext
windows): the row quantizer runs first, the codec sees the dequantized rows, and
the byte count is the codec archive alone — the quantized values are replaced,
not stored twice.

| stack | ppl | Δ vs f16 | of fp16 |
|---|---|---|---|
| f16 | 27.9690 | — | 100% |
| `q8_0` K+V | 28.2485 | +1.00% | 53.12% |
| `q8_0` + dct b32/k32 int8 on K | 28.4259 | +1.63% | 52.34% |
| `q8_0` + dct b8/k8 int8 on K | 28.5795 | +2.18% | 52.34% |
| `q8_0` + dct b8/k8 int8 on K+V | 28.6244 | +2.34% | 51.56% |
| `q8_0` + dct b8/k4 on K | 160.59 | +474% | 51.56% |
| `q8_0` + dct b8/k2 on K | 2763.8 | +9781% | 39.06% |
| `q4_0` K+V | 201.95 | +622% | 28.12% |
| `q4_0` + dct b8/k8 int8 on K | 211.70 | +657% | 39.84% |
| `q4_0` + dct b8/k4 on K | 1195.1 | +4173% | 39.06% |
| `q4_0` + dct b8/k2 int8 on K | 5831.6 | +20751% | 20.51% |
| `q4_0` + dct b8/k2 int8 on K+V | 42085 | +150371% | 12.89% |

Reading:

- **On top of `q8_0` the codec can only reclaim the scale overhead** — one
  fp32 scale per 128 coefficient values instead of one fp16 scale per 32 raw
  values. That is 0.78% of the total bytes on K (52.34% vs 53.12%) and 1.56% on
  K+V, while the damage grows 1.6–2.3×. Strictly dominated.
- **On top of `q4_0` it is worse than useless.** An int8 coefficient archive is
  ~1 B/value, i.e. *larger* than the 0.5625 B/value it replaces, so the
  lossless-basis stack actually costs **more** bytes (39.84% vs 28.12%) for a
  worse model. Only aggressive truncation gets back under 4 bits, and those rows
  (20.5% at ppl 5831, 12.9% at ppl 42085) are broken models, not compressions.
- **The two stages are not complementary.** Both spend the same budget on the
  same information (K), and llama.cpp's per-32 absmax is already the cheaper way
  to spend it; putting an orthonormal transform in front only moves the error
  around and adds a second source of it.

The base quantizer here is an emulation in bf16 (symmetric absmax per 32 head
dims, applied to K and V), so its absolute damage is harsher than the real
`-ctk q8_0` (+1.0% vs +0.05% in §1: the emulation also quantizes V, which the
real configuration does not). The ordering, and every byte count, is what the
argument rests on.

## 4. Why the codec loses

1. **Truncation deletes variance, quantization only adds noise.** `dct k=4`
   (half the coefficients kept, 75% of fp16 K bytes) damages as much as
   `q4_0` (28% of fp16 K+V bytes). Attention logits are far more sensitive to a
   missing subspace than to a bounded perturbation. `k=2` is not a compression
   trade-off, it is a broken model. This matches the dense-model rows already in
   the README's perplexity table (16.8 → 470 → 6486), so it is not a
   llama.cpp-specific artifact.
2. **The gain from the basis is "decorrelate, then quantize" — and row
   quantization already has it.** Enlarging the block monotonically helps
   (b8 +1.29% → b16 +0.54% → b32 +0.38%), which is the signature of
   decorrelation, not of truncation. But ggml's per-32 absmax scale adapts
   locally every 32 values, while this codec stores one scale per
   (block, frequency) — i.e. one per 128 values. Larger blocks converge toward a
   global transform; they do not overtake the per-32 scale.
3. **K and V are not interchangeable.** V tolerates 4-bit almost for free; K
   does not even tolerate it at 8-bit-plus-margin. Compressing both with the
   same codec (which is what `FlashCodecCache` does) is therefore the wrong
   shape of design. This is the same conclusion that llama.cpp's DeepSeek-V4
   cache encodes structurally: `llama_kv_cache_dsv4` keeps a **compressed K-only**
   tier and leaves V alone.

## 5. What this means for flashcodec

- The library's value is **storage-side research on the sequence axis**, not a
  production KV-cache replacement for llama.cpp. The honest framing is the one
  already in *Boundaries*: storage-layer gains, decoded views materialized for
  attention.
- Competing at 50% bytes is pointless: `q8_0` is free there. Any future codec
  work has to target **K at 25% bytes**, and the data says the way to get there
  is a rotation / per-channel quantizer along the head dim (llama.cpp already
  ships Hadamard-transform machinery for this), not coefficient truncation.
- For users of this project who deploy on llama.cpp, the measured takeaway is
  simply: `-ctk q8_0 -ctv q4_0` (40.6% of fp16 KV, +0.28% ppl), never
  `-ctk q4_0`.

## 6. Limitations

- One model (Qwen3-0.6B), one context length (512). Longer contexts usually make
  KV quantization *more* damaging, but they should not reorder DCT vs row
  quantization, because the two have different error mechanisms.
- HF bf16 vs llama.cpp fp16; `q8_0 emul` is a semantic emulation, not bit-exact.
- Hybrid models (Qwen3.5-0.8B) are not re-measured here: only 6 of 24 layers
  carry a codec there, so the absolute damage is small, but at equal bytes those
  six layers are still better served by `q8_0`.
- Capacity: at 0.6B and 128k context, `K=q8_0 + V=q4_0` needs only ~4.7 GB, so
  there is no VRAM pressure on a 32 GB card. The capacity argument only bites
  for large models at very long context — where `q8_0` K + `q4_0` V is still the
  cheaper answer.

## 7. Reproduce

```bash
# convert (needs the `gguf` + `sentencepiece` python packages)
python llama.cpp/convert_hf_to_gguf.py <qwen3-0.6b-hf-dir> \
    --outfile qwen3-0.6b-f16.gguf --outtype f16

# llama.cpp matrix
llama.cpp/build/bin/llama-perplexity -m qwen3-0.6b-f16.gguf \
    -f wikitext-2-raw/wiki.test.raw -c 512 --chunks 40 \
    -ngl 99 -fa on -ctk q8_0 -ctv q4_0

llama.cpp/build/bin/llama-bench -m qwen3-0.6b-f16.gguf \
    -p 4096 -n 32 -fa 1 -ngl 99 -ctk q8_0 -ctv q8_0 -r 3
```

The HF-side A/B harness (K-only / V in {raw, quant, codec}, ggml-style row
quantization emulation, DCT tiers) is a standalone script kept with the
experiment record; it is not part of the published package.
