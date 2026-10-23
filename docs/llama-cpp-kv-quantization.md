# Same-bytes comparison with llama.cpp KV quantization

Measured 2026-10-23 on **Qwen3-0.6B** (28 layers, 8 KV heads, head_dim 128, i.e.
112 KiB of KV per token in fp16), RTX 5090 D, llama.cpp `184f65360`
(CUDA + flash-attention enabled), wikitext-2-raw.

## TL;DR

llama.cpp's own `-ctk q8_0` reaches **~53% of fp16 KV bytes at +0.05% perplexity**.
Two usable flashcodec **archive** tiers lose against it at equal bytes:

| flashcodec tier | bytes | damage at equal bytes vs `q8_0` |
|---|---|---|
| truncation (`k < r`) | 25–75% | **2.7× more bytes for the same damage**; `k=2` is unusable |
| lossless basis + `quant_bits=8` | 50% | **8× more damage** (+0.38% vs +0.05%) |

Stacking the codec on top of llama.cpp's quantizer does not help either: on
`q8_0` it reclaims 0.8% of the bytes for 63% more damage (§3).

**But the loss is the coefficient quantizer, not the transform.** The archive
above stores one fp32 scale per frequency row (128 values). Give the coefficients
the same per-32 absmax that ggml uses — identical bytes — and the ordering flips:
**5× less perplexity damage at 8 bits and 7× less at 4 bits (§5)**, with the
*reconstruction error unchanged*. The gain is error shaping (the error lands in
the slowly-varying subspace, which softmax barely sees), not a better K. That
result is HF-measured and still needs confirming inside llama.cpp; it does not
make 4-bit K usable on this model (ppl 41.6 vs 18.6).

**Scale check.** On a 27B hybrid (`qwen35`, 17 KV-bearing layers) the same
`q4_0` KV config costs +0.39% at 8k context and +0.20% at 32k — the small-model
cliff (+236% on 0.6B) does not exist at production scale (§6), and the 4-bit KL
there is 0.0035, so there is nothing to fix. On the 0.6B it is already broken.
Both ends leave the transform without a job; the dense mid-size case below is
the only remaining opening.

**The one place it could matter** is a dense mid-size model, where 4-bit is
costly but not fatal: Gemma4-12B (48 layers, SWA + 8 global MQA layers) shows
KL 0.30 at 4 bits vs 0.17 at 8 bits with top-1 agreement 89% (§7) — 86× more
distortion than the 27B hybrid, because every layer reads the quantized cache.
Perplexity is unusable there (four corpora give baselines of 302 / 355 / 2.24 /
18.87 and the sign of the effect flips), so that section is measured with
`--kl-divergence` instead.

Finally, K and V are not equally sensitive, which changes how a KV budget should
be spent.

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

## 5. Where the transform does win: the same quantizer on coefficients

§2–§4 tested flashcodec's *own* archive format, whose coefficient quantizer
stores one fp32 scale per (block, frequency) row, i.e. one per 128 values. That
is coarser than ggml's per-32 absmax. Feeding the same per-32 quantizer to the
coefficients instead (lossless basis, `k = r`, so the byte count is **identical**
to the same ggml type on raw rows) flips the result:

fp16 model, 1024-token windows scored on the last 512 (baseline ppl 18.59, close
to llama.cpp's 19.60):

| tier | ppl | Δ | of fp16 |
|---|---|---|---|
| f16 | 18.5913 | — | 100% |
| `q8_0` K+V (raw rows) | 18.7294 | +0.1382 | 53.12% |
| dct b8 + q8 K, q8 V | 18.6501 | +0.0588 | 53.12% |
| dct b16 + q8 K, q8 V | 18.6282 | +0.0369 | 53.12% |
| **dct b32 + q8 K, q8 V** | **18.6192** | **+0.0279** | **53.12%** |
| `q4_0` K+V (raw rows) | 186.83 | +168.24 | 28.12% |
| dct b16 + q4 K, q4 V | 64.33 | +45.74 | 28.12% |
| dct b32 + q4 K, q4 V | 47.83 | +29.24 | 28.12% |
| dct b32 + q4 K, q4 V (K only dct) | 41.61 | +23.01 | 28.12% |

At **identical bytes**: 5× less perplexity damage at 8 bits, and 7× less at
4 bits. Replicated on disjoint 512-token windows (`q8_0` +0.3595 vs
dct b32 +0.0536; `q4_0` 196.3 vs 46.8). Block 32 > 16 > 8 in both regimes.

### It is error shaping, not better reconstruction

Measured on real K rows (Qwen3-0.6B, layer 0/6/12/18/24), the **reconstruction
error is the same** for both quantizers:

| variant | rel-L2 | max abs err |
|---|---|---|
| raw q8 | 0.0111–0.0124 | 0.17–1.70 |
| dct b32 + q8 | 0.0110–0.0118 | 0.19–1.65 |

So the transform does not preserve K better; it moves the quantization error
into the low-frequency (slowly varying along positions) subspace. A logit offset
that is nearly constant across neighbouring positions is close to invisible to
softmax — only the error that changes the *relative* ordering of positions
costs perplexity. Hence the same L2 error and a much smaller ppl cost, and hence
also why larger blocks (finer frequency resolution) keep improving the result.

### What this does and does not change

- It **does not** rescue 4-bit: ppl 41.6 vs 18.6 is still a broken model, even
  though the damage is 7× smaller than `q4_0`.
- It **does** mean the seq-axis transform is a legitimate quantizer front-end
  (coefficient-domain steps matched to frequency), not just a truncation scheme.
  At 8 bits the absolute headroom is small because `q8_0` is already nearly
  free, so the value only shows up in regimes where the byte budget is tighter.
- It is measured in HF, not in llama.cpp, and this harness's absolute damage is
  larger than llama.cpp's real `q8_0` (+0.74% vs +0.05% at a comparable
  baseline). A llama.cpp implementation is needed before quoting it as a win
  there; a larger model is needed before claiming it makes 4-bit K usable.

## 6. Model size: the cliff is a small-model property

The same KV-type sweep on a production-scale local checkpoint — DrakIdol
Qwen3.8-27B-Plus, GGUF `Q4_K`, arch `qwen35`, 64 layers with
`full_attention_interval = 4` (17 KV-bearing layers), 4 KV heads × 256 dim,
i.e. 68 KiB of KV per token:

| context | f16 | `q8_0` K+V | `q4_0` K+V |
|---|---|---|---|
| 512 (20 chunks) | 6.8003 | 6.8038 (+0.05%) | 6.8130 (+0.19%) |
| 8192 (4 chunks) | 5.4791 | — | 5.5005 (+0.39%) |
| 32768 (4 chunks) | 6.1354 | 6.1384 (+0.05%) | 6.1475 (+0.20%) |

At 8k the split is K-only +0.24%, V-only +0.13%, and `q8_0` K + `q4_0` V is
+0.06% — all inside the ±0.10 CI. So the 4-bit regime that collapses the 0.6B
(+236%) is **essentially free at 27B**, and stays free out to 32k context.

Consequence: the cliff is a property of small models, not of KV quantization.
The DCT win of §5 sits exactly where the model is already broken, and the models
that actually have a byte-budget problem (large, long context) do not need it.
That leaves §5 as a mechanism result — the error shaping is real and reproducible
— rather than a deployment feature.

(Locally there is also Gemma4-12B, dense with 48 layers, but on wikitext its
perplexity is ~302 because it is a roleplay finetune; its deltas are dominated by
the ±23 confidence interval — e.g. `q4_0` K+V +13%, single-side +4%/0%. A
decisive dense mid-size test needs an in-domain corpus and a distribution metric
(KL or token agreement) rather than perplexity.)

## 7. Gemma 4 12B: a dense/SWA model, and why perplexity is the wrong metric

Local checkpoint: Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced, GGUF `Q4_K_M`,
arch `gemma4`, 48 layers = **40 sliding-window layers** (window 1024, 8 KV heads
x 256 dim) + **8 global layers** (1 KV head x 512 dim), 262k vocab.

Its perplexity is unusable as a measurement: the same quantized tiers give
baseline 302 on wikitext, 355 on a human-written chat corpus, 2.24 on text the
model generated itself, and 18.87 on a second self-generated corpus — and in the
last case *quantizing improves* ppl (18.87 -> 8.84 for `q4_0` K+V), because the
noise flattens an over-confident distribution. Heavily finetuned models are
calibrated to their own domain, so ppl measures calibration, not distortion.

Use `llama-perplexity --kl-divergence --save-all-logits` instead (mean KL of the
quantized distribution against the f16 one, plus top-1 agreement). Corpus =
20k tokens generated by the model itself, c=2048 x 4 chunks (8192 scored):

| KV | mean KL | ppl ratio | same top-1 |
|---|---|---|---|
| f16 (sanity) | 0.000000 | 1.006 | 100.0% |
| `q8_0` K+V | 0.1678 | 1.034 | 93.4% |
| `q4_0` K, f16 V | 0.2295 | 1.056 | 90.6% |
| f16 K, `q4_0` V | 0.2247 | 1.123 | 90.8% |
| `q4_0` K+V | 0.3027 | 1.127 | 89.1% |

Reading:

- **No cliff at 12B**, but 4-bit is not free either: KL 0.30 vs 0.17 at 8 bits,
  and top-1 agreement drops to 89%.
- **K and V contribute equally in KL** (0.2295 vs 0.2247) even though ppl calls V
  twice as bad (+12.3% vs +5.6%). ppl and distribution distance disagree about
  which side matters; only the latter is a reliable target.
- The f16 sanity row also fixes the run noise: KL is exactly 0, while the ppl
  ratio drifts by 0.6% between identical runs — another reason to report KL.

For comparison, the same KL protocol on the 27B hybrid (wikitext, c=2048 x 4):

| KV | mean KL | ppl ratio | same top-1 |
|---|---|---|---|
| `q8_0` K+V | 0.0021 | 0.997 | 97.8% |
| `q4_0` K+V | 0.0035 | 1.000 | 96.9% |

So the 27B distorts **~86x less** than the 12B at 4 bits. That is not a
parameter-count effect but an **exposure** effect: in `qwen35` only 17 of 64
layers have a KV cache (the rest are linear attention with recurrent state that
`-ctk/-ctv` never touches), while in `gemma4` every layer's attention reads the
quantized cache. Dense small-to-mid models are therefore the only place where
KV compression is both wanted and measurably costly — and at 12B the cost of
4-bit is a KL of 0.30, not a collapse.

Capacity is still not the motivator here: Gemma4-12B keeps ~32 KiB of KV per
token in its 8 global layers and its 40 SWA layers are capped at the 1024-token
window, so 8k context costs ~0.9 GiB — there is no byte pressure to relieve.

## 8. What this means for flashcodec

- The seq-axis transform is a **real distortion win at equal bytes** (5× at
  8 bits, 7× at 4 bits, §5), but as a *coefficient-domain* transform with
  ggml-style per-32 scales, not as this library's archive format, and it is
  still HF-measured only.
- The only defensible target is a **dense mid-size model**: Gemma4-12B is where
  4-bit KV first costs something real (KL 0.30, top-1 agreement 89%, §7) without
  being a collapse. Applying the §5 transform to that regime is the one
  experiment that could turn this into a feature; hybrid models (27B) do not
  need it, and 0.6B is already broken at that ratio.
- **Metric discipline**: use KL / top-1 agreement (`--kl-divergence`) for
  finetuned or dense models — their ppl is a calibration artifact and sometimes
  *improves* under quantization. ppl is only informative on base models with a
  sane baseline (the 0.6B and 27B rows).
- Competing at 50% bytes remains pointless (`q8_0` is free there). For
  deployment advice, the measured takeaway is `-ctk q8_0 -ctv q4_0` on base
  models (40.6% of fp16 KV, +0.28% ppl), never `-ctk q4_0` on a small dense
  model.

## 9. Limitations

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

## 10. Reproduce

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
