"""实战测试：Qwen3.5-0.8B（混合线性注意力架构）× flashcodec。

- PNG keep=1.0 无损档：生成 token 与未打补丁基线逐位完全一致（终极正确性）
- JPEG k=2/k=4 有损档：文本流畅度、缓存存储比、与基线的 token 一致率
- 混合架构审计：只有 full_attention 层挂 codec，线性注意力层保留原生层
用法：python benchmarks/run_qwen_demo.py
（模型默认 "Qwen/Qwen3.5-0.8B"，从本地 HF 缓存离线加载；可用
 FLASHCODEC_MODEL / FLASHCODEC_NEW_TOKENS 环境变量覆盖）
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
PROMPT = "杭州的西湖自古以来就是文人墨客向往的地方，"
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


def report(tag, out, cache=None):
    text = tok.decode(out[0], skip_special_tokens=True)
    print(f"[{tag}] {text!r}")
    if cache is not None:
        n_codec = len(cache.codec_layer_indices())
        st = cache.storage_bytes()
        # raw 基准：仅 codec 层的完整 KV（B,H,S,D × key+value）
        l0 = cache.layers[cache.codec_layer_indices()[0]]
        raw = n_codec * l0.keys.numel() * l0.keys.element_size() * 2
        print(f"       seq={cache.get_seq_length()} codec_layers={n_codec}/{n_layers} "
              f"storage={st/1e6:.2f}MB raw={raw/1e6:.2f}MB "
              f"ratio={st/raw:.2%} | decode {dt[0]:.1f}s")
    return out


dt = [0.0]
with warnings.catch_warnings(record=True) as wlist:
    warnings.simplefilter("always")

    # ---- 1) 基线（未打补丁）----
    t0 = time.time()
    with torch.no_grad():
        ref = model.generate(input_ids=ids, **gen_kw)
    dt[0] = time.time() - t0
    report("baseline", ref)

    # ---- 2) PNG 无损档：逐位一致性（终极正确性检验）----
    apply_flash_codec(model, mode="png", block_size=8, keep_ratio=1.0)
    cache = model._flashcodec["caches"]
    t0 = time.time()
    with torch.no_grad():
        out_png = model.generate(input_ids=ids, **gen_kw)
    dt[0] = time.time() - t0
    cache = model._flashcodec["caches"][-1]
    report("png keep=1.0", out_png, cache)
    exact = torch.equal(ref, out_png)
    print(f"       ✅ 生成 token 与基线逐位一致: {exact}"
          if exact else "       ❌ 生成 token 与基线不一致！")
    assert exact, "PNG 无损档位应与基线逐位一致"
    assert cache.codec_layer_indices() == [
        i for i, t in enumerate(
            model.config.get_text_config().layer_types)
        if t == "full_attention"], "只有 full_attention 层应挂 codec"
    remove_flash_codec(model)

    # ---- 3) JPEG 有损档：流畅度 + 存储比 + token 一致率 ----
    for k in (2, 4):
        apply_flash_codec(model, mode="jpeg", block_size=8, keep_freqs=k)
        t0 = time.time()
        with torch.no_grad():
            out_j = model.generate(input_ids=ids, **gen_kw)
        dt[0] = time.time() - t0
        cache = model._flashcodec["caches"][-1]
        report(f"jpeg k={k}", out_j, cache)
        same = (out_j == ref).float().mean().item()
        print(f"       token 一致率 vs 基线: {same:.1%}")
        remove_flash_codec(model)

if wlist:
    print(f"\n⚠️ 捕获到 {len(wlist)} 条警告：")
    for w in wlist[:5]:
        print(f"   {w.category.__name__}: {str(w.message)[:120]}")
else:
    print("\n零警告 ✅")
print("\nDEMO ALL PASS")
