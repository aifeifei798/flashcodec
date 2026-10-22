"""Per-layer budget calibration: is a non-uniform `keep_freqs` schedule worth it?

Procedure (all measured, nothing assumed):
1. baseline with the lossless DCT basis on every layer (k = r, fp);
2. per-layer sensitivity: compress **one** layer to k=4, measure the perplexity
   damage (this is the calibrating signal);
3. build a schedule: the cheapest half of the layers get k=4, the rest stay at
   k=r;
4. compare against uniform k=6 (same average storage), uniform k=4, and the
   int8 reference (k=r + quant_bits=8, half the bf16 storage).

The result on Qwen3-0.6B is a *negative* one — see the README table: the
per-layer damages are super-additive, so calibration buys little and int8 wins.
The script is kept because the calibration is model-dependent; run it on your
own model before trusting any schedule.

Usage: python benchmarks/bench_adaptive.py
Environment: FLASHCODEC_MODEL, FLASHCODEC_CORPUS, FLASHCODEC_PPL_WINDOWS,
             FLASHCODEC_PPL_SEQ (same as bench_ppl.py)
"""

import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from flashcodec import FlashCodecCache

MODEL = os.environ.get("FLASHCODEC_MODEL", "Qwen/Qwen3-0.6B")
CORPUS = os.environ.get(
    "FLASHCODEC_CORPUS",
    str(Path(__file__).resolve().parents[2] /
        "minimind_dataset/pretrain_t2t_mini.jsonl"))
N_WINDOWS = int(os.environ.get("FLASHCODEC_PPL_WINDOWS", "6"))
SEQ = int(os.environ.get("FLASHCODEC_PPL_SEQ", "512"))
R = 8
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"model={MODEL}  device={device}  windows={N_WINDOWS} x {SEQ}\n")
tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
m = AutoModelForCausalLM.from_pretrained(
    MODEL, local_files_only=True,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
).to(device).eval()


def load_windows(path, n, seq):
    windows, buf = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = doc.get("text") or doc.get("content") or ""
            if not text:
                continue
            buf.extend(tok(text, add_special_tokens=False).input_ids)
            buf.append(tok.eos_token_id)
            while len(buf) >= seq:
                windows.append(torch.tensor(buf[:seq], device=device).unsqueeze(0))
                buf = buf[seq:]
                if len(windows) >= n:
                    return windows
    return windows


windows = load_windows(CORPUS, N_WINDOWS, SEQ)
full = [i for i, t in enumerate(m.config.get_text_config().layer_types)
        if t == "full_attention"]
print(f"windows={len(windows)}  full_attention_layers={len(full)}\n")


def eval_ppl(per_layer=None, **base_cfg):
    cfg = dict(mode="jpeg", block_size=R, keep_freqs=R)
    cfg.update(base_cfg)
    total_nll, total_tok = 0.0, 0
    for ids in windows:
        cache = FlashCodecCache.from_config(m.config, per_layer=per_layer, **cfg)
        with torch.no_grad():
            out = m(input_ids=ids, labels=ids, use_cache=True,
                    past_key_values=cache)
        total_nll += out.loss.item() * ids.shape[1]
        total_tok += ids.shape[1]
    return math.exp(total_nll / total_tok)


base = eval_ppl()
print(f"baseline (k=r, fp)            ppl {base:.4f}")

sens = {}
for li in full:
    p = eval_ppl(per_layer={li: dict(keep_freqs=4)})
    sens[li] = p - base
    print(f"  layer {li:>2}: k=4 damage {sens[li]:+.4f}")

order = sorted(full, key=lambda i: sens[i])
half = len(full) // 2
sched = {i: dict(keep_freqs=4) for i in order[:half]}

rows = [
    ("uniform k=6 (75%)", eval_ppl(keep_freqs=6), dict(keep_freqs=6), 6 / R),
    ("uniform k=4 (50%)", eval_ppl(keep_freqs=4), dict(keep_freqs=4), 4 / R),
    (f"adaptive ({half} layers k=4)", eval_ppl(per_layer=sched),
     None, (half * 4 + (len(full) - half) * R) / (len(full) * R)),
    ("uniform k=r int8", eval_ppl(quant_bits=8),
     dict(quant_bits=8), 0.5),
]

print(f"\n{'schedule':<26} | {'ppl':>9} | {'Δppl':>9} | {'storage':>8}")
print("-" * 60)
for name, ppl, _, ratio in rows:
    print(f"{name:<26} | {ppl:>9.4f} | {ppl - base:>+9.4f} | {ratio:>7.2%}")
print(f"\ncalibrated order (cheapest first): {order[:half]}")
print("storage is relative to the same fp archive at k=r (int8 counts as 0.5x).")
