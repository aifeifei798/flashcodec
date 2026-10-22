"""End-to-end decode benchmark: raw DynamicCache vs codec cache, with and
without the materialization-free attention patch.

Long-context decode (1024-token prompt + 64 new tokens) on a dense model:
reports tokens/s, peak memory during generate, and the memory still allocated
after generate (steady state). With `materialization_free=True` the prefill view
is dropped and decode attends on the stored rows, so the steady state should be
the archive only.

Usage:
    python benchmarks/bench_e2e.py
    FLASHCODEC_MODEL=Qwen/Qwen3-0.6B FLASHCODEC_NEW_TOKENS=64 \
        FLASHCODEC_PROMPT_TOKENS=1024 python benchmarks/bench_e2e.py
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from flashcodec import apply_flash_codec, remove_flash_codec

MODEL = os.environ.get("FLASHCODEC_MODEL", "Qwen/Qwen3-0.6B")
CORPUS = os.environ.get(
    "FLASHCODEC_CORPUS",
    str(Path(__file__).resolve().parents[2] /
        "minimind_dataset/pretrain_t2t_mini.jsonl"))
NEW = int(os.environ.get("FLASHCODEC_NEW_TOKENS", "64"))
PROMPT_TOKENS = int(os.environ.get("FLASHCODEC_PROMPT_TOKENS", "1024"))
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"model={MODEL}  device={device}  prompt={PROMPT_TOKENS}  new={NEW}\n")
tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, local_files_only=True,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
).to(device).eval()

ids = []
with open(CORPUS, encoding="utf-8") as fh:
    for line in fh:
        doc = json.loads(line)
        ids.extend(tok(doc.get("text", ""), add_special_tokens=False).input_ids)
        if len(ids) >= PROMPT_TOKENS:
            break
ids = torch.tensor(ids[:PROMPT_TOKENS], device=device).unsqueeze(0)
print(f"prompt shape {tuple(ids.shape)}\n")

GEN = dict(max_new_tokens=NEW, min_new_tokens=NEW, do_sample=False,
           pad_token_id=tok.eos_token_id)


def measure(tag, **patch):
    if patch:
        apply_flash_codec(model, **patch)
        # warm up after patching: the Triton kernel compiles on first use
        with torch.no_grad():
            model.generate(input_ids=ids[:, :256], max_new_tokens=2,
                           do_sample=False, pad_token_id=tok.eos_token_id)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated()
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(input_ids=ids, **GEN)
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    if device == "cuda":
        peak = (torch.cuda.max_memory_allocated() - base) / 1e6
        steady = (torch.cuda.memory_allocated() - base) / 1e6
    else:
        peak = steady = float("nan")
    cache = (model._flashcodec["caches"][-1]
             if patch and model._flashcodec["caches"] else None)
    storage = cache.storage_bytes() / 1e6 if cache is not None else 0.0
    wrapped = model._flashcodec.get("codec_attention_modules", 0) if patch else 0
    remove_flash_codec(model) if patch else None
    print(f"{tag:<28} | {NEW / dt:>6.1f} tok/s | peak {peak:>6.0f}MB | "
          f"steady {steady:>6.0f}MB | archive {storage:>6.2f}MB"
          + (f" | wrapped {wrapped}" if patch else ""))
    return out


# warm up kernels
with torch.no_grad():
    model.generate(input_ids=ids[:, :64], max_new_tokens=2, do_sample=False,
                   pad_token_id=tok.eos_token_id)

print(f"{'tier':<28} | {'speed':>10} | {'peak':>12} | {'steady':>13} | {'storage':>13}")
print("-" * 92)
ref = measure("baseline (raw KV)")
measure("codec jpeg k=8 int8", mode="jpeg", block_size=8, keep_freqs=8,
        quant_bits=8)
measure("codec jpeg k=2 int8", mode="jpeg", block_size=8, keep_freqs=2,
        quant_bits=8)
measure("codec k=2 int8 materialization-free", mode="jpeg", block_size=8,
        keep_freqs=2, quant_bits=8, materialization_free=True)

print("\npeak = max allocated during generate (prefill dominates); steady = still\n"
      "allocated afterwards. materialization-free drops the prefill view, so the\n"
      "steady state becomes the archive (plus any dequant temporaries).")
