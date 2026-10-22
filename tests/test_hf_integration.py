"""transformers 5.x 原生集成测试（离线构造微型 Llama，无网络依赖）：
A) 显式传参：FlashCodecCache 直接传给 model.generate
B) 热补丁：apply_flash_codec 后 generate 自动注入；remove_flash_codec 还原
C) PNG 无损：生成 token 序列与未打补丁模型 100% 逐位相同
D) 长文本稳定性：70+ token 跨 8+ 个块，多轮归档无崩溃/shape 异常，
   且存储占用符合压缩比（无泄漏）
运行：python tests/test_hf_integration.py（或 pytest）
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from flashcodec import (FlashCodecCache, apply_flash_codec,
                        find_attention_modules, remove_flash_codec)

torch.manual_seed(0)
VOCAB, PROMPT_LEN, NEW = 128, 12, 20


def build_tiny_llama(seed=7):
    """离线构造微型 Llama（2 层，hidden 32）。放大 lm_head 使 logits 间隔
    明确（~×20），隔离随机平局抖动，专注检验缓存机制正确性。"""
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=VOCAB, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=2,
                      num_key_value_heads=2, max_position_embeddings=256)
    m = LlamaForCausalLM(cfg).eval()
    m.generation_config.eos_token_id = None     # 随机权重会误吐 eos，关闭提前停止
    with torch.no_grad():
        m.lm_head.weight.mul_(20.0)
    return m


def gen(model, **kw):
    kw.setdefault("max_new_tokens", NEW)
    with torch.no_grad():
        return model.generate(input_ids=PROMPT(), do_sample=False,
                              pad_token_id=0, **kw)


def PROMPT():
    g = torch.Generator().manual_seed(42)
    return torch.randint(0, VOCAB, (1, PROMPT_LEN), generator=g)


def test_a_explicit_cache():
    model = build_tiny_llama()
    cache = FlashCodecCache(mode="jpeg", block_size=8, keep_freqs=2)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)   # 零警告验收
        out = gen(model, past_key_values=cache)
    assert out.shape == (1, PROMPT_LEN + NEW), f"输出形状异常 {out.shape}"
    # 末位生成 token 的 KV 从不被计算（无人 attend），故 cache 长度 = 总长-1
    assert cache.get_seq_length() == PROMPT_LEN + NEW - 1, \
        f"seq_length={cache.get_seq_length()} != {PROMPT_LEN + NEW - 1}"
    assert cache._seen_tokens == PROMPT_LEN + NEW - 1
    assert len(cache.layers) == model.config.num_hidden_layers, \
        f"层数异常 {len(cache.layers)}"
    # jpeg 归档生效：存储显著小于 raw（k/r = 2/8 + 尾部 raw）
    raw = (PROMPT_LEN + NEW) * 1 * 2 * 32 * 4 * 2
    assert cache.storage_bytes() < 0.5 * raw, "jpeg 压缩未生效"


def test_b_hotpatch():
    model = build_tiny_llama()
    assert len(find_attention_modules(model)) == model.config.num_hidden_layers
    apply_flash_codec(model, mode="jpeg", block_size=8, keep_freqs=2)
    out = gen(model)
    assert out.shape == (1, PROMPT_LEN + NEW)
    caches = model._flashcodec["caches"]
    assert len(caches) == 1 and isinstance(caches[0], FlashCodecCache)
    assert caches[0].get_seq_length() == PROMPT_LEN + NEW - 1
    remove_flash_codec(model)
    assert model._flashcodec is None
    out2 = gen(model)                                 # 还原后正常生成
    assert out2.shape == (1, PROMPT_LEN + NEW)


def test_c_png_lossless_exact():
    """keep_ratio=1.0 无损档位：生成序列与原模型逐 token 完全一致。"""
    base = build_tiny_llama()
    ref = gen(base)
    apply_flash_codec(base, mode="png", block_size=8, keep_ratio=1.0)
    out = gen(base)
    assert torch.equal(ref, out), (
        f"PNG 无损档位生成不一致: ref={ref[0, -NEW:].tolist()} "
        f"got={out[0, -NEW:].tolist()}")


def test_d_long_stability():
    """70 token（>8 个 8-token 块）：多轮归档无异常，存储符合压缩比。"""
    long_new = 70
    for mode, kw, expect in (("jpeg", dict(keep_freqs=2), 0.35),
                             ("png", dict(keep_ratio=1.0), 1.05)):
        model = build_tiny_llama()
        cache = FlashCodecCache(mode=mode, block_size=8, **kw)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            out = gen(model, past_key_values=cache, max_new_tokens=long_new)
        total = PROMPT_LEN + long_new
        kv_len = total - 1                        # 末位 token 的 KV 不被计算
        assert out.shape == (1, total), f"{mode} 长文本形状异常"
        assert cache.get_seq_length() == kv_len
        layer0 = cache.layers[0]
        assert layer0._key_stream.n_archived > 0, f"{mode} 未发生归档"
        assert kv_len - layer0._key_stream.n_archived < 8, "raw 尾应不足一块"
        raw = kv_len * 1 * 2 * 32 * 4 * 2
        assert cache.storage_bytes() < expect * raw, \
            f"{mode} 存储未压缩: {cache.storage_bytes()} vs raw {raw}"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = True
    for fn in fns:
        try:
            fn()
            print(f"✅ {fn.__name__}")
        except AssertionError as e:
            ok = False
            print(f"❌ {fn.__name__}: {e}")
    import gc
    gc.collect()
    print("\nALL PASS" if ok else "\nFAILURES")
    sys.exit(0 if ok else 1)
