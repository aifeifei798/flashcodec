"""End-to-end test: Qwen3.5-0.8B (hybrid linear attention) x flashcodec.

- PNG keep=1.0 lossless mode: generated tokens are bitwise-identical to the
  unpatched baseline (ultimate correctness check)
- JPEG k=2/k=4 lossy modes: text fluency, cache storage ratio, and token
  agreement rate against the baseline
- Hybrid-architecture audit: only full_attention layers get a codec; linear
  attention layers keep their native layers
Usage: python benchmarks/run_qwen_demo.py
(Model defaults to "Qwen/Qwen3.5-0.8B", loaded offline from the local HF cache;
override with the FLASHCODEC_MODEL / FLASHCODEC_NEW_TOKENS env variables)
"""

import os
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from flashcodec import (FlashCodecCache, apply_flash_codec,
                        remove_flash_codec)

MODEL = os.environ.get("FLASHCODEC_MODEL", "Qwen/Qwen3.5-0.8B")
NEW = int(os.environ.get("FLASHCODEC_NEW_TOKENS", "48"))
PROMPT = "Hangzhou's West Lake has drawn poets and scholars since ancient times, "
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"model={MODEL}  device={device}  new_tokens={NEW}\n")
tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
dtype = torch.bfloat16 if device == "cuda" else torch.float32
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, torch_dtype=dtype)
except Exception:
    from transformers import AutoModelForImageTextToText
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, local_files_only=True, torch_dtype=dtype)
model = model.to(device).eval()
n_layers = model.config.get_text_config().num_hidden_layers

ids = tok(PROMPT, return_tensors="pt").input_ids.to(device)
gen_kw = dict(max_new_tokens=NEW, min_new_tokens=NEW, do_sample=False,
              pad_token_id=tok.eos_token_id)


def report(tag, out, cache=None, secs=None, peak=None):
    text = tok.decode(out[0], skip_special_tokens=True)
    print(f"[{tag}] {text!r}")
    if cache is not None:
        n_codec = len(cache.codec_layer_indices())
        st = cache.storage_bytes()
        # raw baseline: full KV of the codec layers only (B,H,S,D x key+value)
        l0 = cache.layers[cache.codec_layer_indices()[0]]
        raw = n_codec * l0.keys.numel() * l0.keys.element_size() * 2
        extra = (f" | {NEW / secs:.1f} tok/s" if secs else "")
        print(f"       seq={cache.get_seq_length()} codec_layers={n_codec}/{n_layers} "
              f"storage={st/1e6:.2f}MB raw={raw/1e6:.2f}MB "
              f"ratio={st/raw:.2%}{extra}"
              + (f" peak=+{peak:.0f}MB" if peak is not None else ""))
    elif secs:
        print(f"       {NEW / secs:.1f} tok/s"
              + (f" peak=+{peak:.0f}MB" if peak is not None else ""))
    return out


def gen_timed(**kw):
    """Generate with a fair timing: synchronize (CUDA is async) and report the
    peak memory allocated on top of the model weights."""
    if device == "cuda":
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(input_ids=ids, **gen_kw, **kw)
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    peak = ((torch.cuda.max_memory_allocated() - base) / 1e6
            if device == "cuda" else 0.0)
    return out, dt, peak


# warm up the kernels so the first timed run is not measuring lazy init
with torch.no_grad():
    model.generate(input_ids=ids, max_new_tokens=2, do_sample=False,
                   pad_token_id=tok.eos_token_id)


with warnings.catch_warnings(record=True) as wlist:
    warnings.simplefilter("always")

    # ---- 1) Baseline (unpatched) ----
    ref, secs, peak = gen_timed()
    report("baseline", ref, secs=secs, peak=peak)

    # ---- 2) PNG lossless mode: bitwise identity (ultimate correctness check) ----
    apply_flash_codec(model, mode="png", block_size=8, keep_ratio=1.0)
    out_png, secs, peak = gen_timed()
    cache = model._flashcodec["caches"][-1]
    report("png keep=1.0", out_png, cache, secs=secs, peak=peak)
    exact = torch.equal(ref, out_png)
    print(f"       ✅ generated tokens bitwise-identical to baseline: {exact}"
          if exact else "       ❌ generated tokens differ from baseline!")
    assert exact, "PNG lossless mode must be bitwise-identical to the baseline"
    assert cache.codec_layer_indices() == [
        i for i, t in enumerate(
            model.config.get_text_config().layer_types)
        if t == "full_attention"], "only full_attention layers should carry a codec"
    remove_flash_codec(model)

    # ---- 3) JPEG lossy mode: fluency + storage ratio + token agreement ----
    tiers = [
        ("jpeg k=8 int8",
         dict(mode="jpeg", block_size=8, keep_freqs=8, quant_bits=8)),
        ("jpeg k=2", dict(mode="jpeg", block_size=8, keep_freqs=2)),
        ("jpeg k=4", dict(mode="jpeg", block_size=8, keep_freqs=4)),
        ("jpeg k=2 int8",
         dict(mode="jpeg", block_size=8, keep_freqs=2, quant_bits=8)),
    ]
    for tag, kw in tiers:
        apply_flash_codec(model, **kw)
        out_j, secs, peak = gen_timed()
        cache = model._flashcodec["caches"][-1]
        report(tag, out_j, cache, secs=secs, peak=peak)
        same = (out_j == ref).float().mean().item()
        print(f"       token agreement vs baseline: {same:.1%}")
        remove_flash_codec(model)

if wlist:
    print(f"\n⚠️ captured {len(wlist)} warnings:")
    for w in wlist[:5]:
        print(f"   {w.category.__name__}: {str(w.message)[:120]}")
else:
    print("\nZero warnings ✅")
print("\nDEMO ALL PASS")
