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
row-wise KV quantization. What the measurements *do* show is that K and V are
not equally sensitive, which changes how a KV budget should be spent.

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

## 3. Why the codec loses

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

## 4. What this means for flashcodec

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

## 5. Limitations

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

## 6. Reproduce

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
