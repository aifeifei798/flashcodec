"""Perplexity benchmark: does KV compression hurt language modelling?

Token agreement (the other benchmark) is a harsh, discrete metric; perplexity
shows how much distribution mass is actually lost. Runs non-overlapping
512-token windows from a local text corpus through a plain forward pass with the
codec cache attached, so the compression is exercised exactly like a prefill.

Usage:
    python benchmarks/bench_ppl.py
Environment:
    FLASHCODEC_MODEL        model id / path (default Qwen/Qwen3.5-0.8B, offline)
    FLASHCODEC_CORPUS       JSONL corpus (default ../minimind_dataset/pretrain_t2t_mini.jsonl)
    FLASHCODEC_PPL_WINDOWS  number of eval windows (default 16)
    FLASHCODEC_PPL_SEQ      window length (default 512)
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

MODEL = os.environ.get("FLASHCODEC_MODEL", "Qwen/Qwen3.5-0.8B")
CORPUS = os.environ.get(
    "FLASHCODEC_CORPUS",
    str(Path(__file__).resolve().parents[2] /
        "minimind_dataset/pretrain_t2t_mini.jsonl"))
N_WINDOWS = int(os.environ.get("FLASHCODEC_PPL_WINDOWS", "16"))
SEQ = int(os.environ.get("FLASHCODEC_PPL_SEQ", "512"))
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"model={MODEL}  device={device}  windows={N_WINDOWS} x {SEQ} tokens\n")
tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
dtype = torch.bfloat16 if device == "cuda" else torch.float32
model = AutoModelForCausalLM.from_pretrained(
    MODEL, local_files_only=True, torch_dtype=dtype).to(device).eval()


def load_windows(path, n_windows, seq):
    """Stream the JSONL corpus, tokenize until enough windows are collected."""
    if not os.path.exists(path):
        sys.exit(f"corpus not found: {path}\n"
                 f"set FLASHCODEC_CORPUS to a JSONL file with a 'text' field")
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
            ids = tok(text, add_special_tokens=False).input_ids
            buf.extend(ids)
            buf.append(tok.eos_token_id)
            while len(buf) >= seq:
                windows.append(torch.tensor(buf[:seq], device=device).unsqueeze(0))
                buf = buf[seq:]
                if len(windows) >= n_windows:
                    return windows
    return windows


windows = load_windows(CORPUS, N_WINDOWS, SEQ)
if not windows:
    sys.exit("no evaluation windows could be built from the corpus")
print(f"corpus={CORPUS}\nwindows={len(windows)} x {SEQ} tokens "
      f"({len(windows) * SEQ} tokens)\n")

TIERS = [
    ("baseline (raw KV)", None),
    ("png keep=1.0", dict(mode="png", block_size=8, keep_ratio=1.0)),
    ("jpeg k=8 (lossless)", dict(mode="jpeg", block_size=8, keep_freqs=8)),
    ("jpeg k=8 int8", dict(mode="jpeg", block_size=8, keep_freqs=8,
                           quant_bits=8)),
    ("jpeg k=4", dict(mode="jpeg", block_size=8, keep_freqs=4)),
    ("jpeg k=2", dict(mode="jpeg", block_size=8, keep_freqs=2)),
    ("jpeg k=2 int8", dict(mode="jpeg", block_size=8, keep_freqs=2,
                           quant_bits=8)),
]


def eval_ppl(cfg):
    total_nll, total_tok, storage, raw = 0.0, 0, 0, 0
    for ids in windows:
        cache = None
        if cfg is not None:
            cache = FlashCodecCache.from_config(model.config, **cfg)
        with torch.no_grad():
            out = model(input_ids=ids, labels=ids, use_cache=cache is not None,
                        past_key_values=cache)
        n = ids.shape[1]
        total_nll += out.loss.item() * n
        total_tok += n
        if cache is not None:
            storage += cache.storage_bytes()
            codec_layers = cache.codec_layer_indices()
            l0 = cache.layers[codec_layers[0]]
            raw += (len(codec_layers) * l0.keys.numel()
                    * l0.keys.element_size() * 2)
    return math.exp(total_nll / total_tok), storage, raw


base_ppl = None
print(f"{'tier':<18} | {'ppl':>10} | {'Δppl':>8} | {'storage':>9}")
print("-" * 54)
for name, cfg in TIERS:
    ppl, storage, raw = eval_ppl(cfg)
    if base_ppl is None:
        base_ppl = ppl
    ratio = f"{storage / raw:>8.2%}" if raw else f"{'1.00':>8}"
    print(f"{name:<18} | {ppl:>10.4f} | {ppl - base_ppl:>+8.4f} | {ratio}")

print("\nnote: png keep=1.0 must match the baseline (lossless tier); the jpeg "
      "rows trade perplexity for archive bytes.")
