"""Native transformers 5.x integration tests (tiny Llama built offline,
no network dependency):
A) Explicit passing: hand a FlashCodecCache straight to model.generate
B) Hot patch: generate auto-injects after apply_flash_codec;
   remove_flash_codec restores the original behaviour
C) PNG lossless: the generated token sequence is 100% bitwise-identical
   to the unpatched model
D) Long-text stability: 70+ tokens spanning 8+ blocks, many archive
   rounds with no crash or shape anomaly, and storage occupancy
   matching the compression ratio (no leaks)
Run: python tests/test_hf_integration.py (or pytest)
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
    """Build a tiny Llama offline (2 layers, hidden 32). Scaling lm_head
    by ~20 keeps the logits margin clear, isolating random tie jitter so
    the test focuses on cache-mechanism correctness."""
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=VOCAB, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=2,
                      num_key_value_heads=2, max_position_embeddings=256)
    m = LlamaForCausalLM(cfg).eval()
    m.generation_config.eos_token_id = None     # random weights emit eos; no early stop
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
        warnings.simplefilter("error", UserWarning)   # zero-warning gate
        out = gen(model, past_key_values=cache)
    assert out.shape == (1, PROMPT_LEN + NEW), \
        f"unexpected output shape {out.shape}"
    # The last generated token's KV is never computed (nobody attends to
    # it), so cache length = total length - 1
    assert cache.get_seq_length() == PROMPT_LEN + NEW - 1, \
        f"seq_length={cache.get_seq_length()} != {PROMPT_LEN + NEW - 1}"
    assert cache._seen_tokens == PROMPT_LEN + NEW - 1
    assert len(cache.layers) == model.config.num_hidden_layers, \
        f"unexpected layer count {len(cache.layers)}"
    # jpeg archiving is active: storage is well below raw
    # (k/r = 2/8 plus the raw tail)
    raw = (PROMPT_LEN + NEW) * 1 * 2 * 32 * 4 * 2
    assert cache.storage_bytes() < 0.5 * raw, "jpeg compression not active"


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
    out2 = gen(model)                                 # works again once restored
    assert out2.shape == (1, PROMPT_LEN + NEW)


def test_c_png_lossless_exact():
    """keep_ratio=1.0 lossless tier: the generated sequence matches the
    original model token for token."""
    base = build_tiny_llama()
    ref = gen(base)
    apply_flash_codec(base, mode="png", block_size=8, keep_ratio=1.0)
    out = gen(base)
    assert torch.equal(ref, out), (
        f"PNG lossless tier diverged: ref={ref[0, -NEW:].tolist()} "
        f"got={out[0, -NEW:].tolist()}")


def test_d_long_stability():
    """70 tokens (>8 blocks of 8): many archive rounds without anomaly,
    storage matching the compression ratio."""
    long_new = 70
    for mode, kw, expect in (("jpeg", dict(keep_freqs=2), 0.35),
                             ("png", dict(keep_ratio=1.0), 1.05)):
        model = build_tiny_llama()
        cache = FlashCodecCache(mode=mode, block_size=8, **kw)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            out = gen(model, past_key_values=cache, max_new_tokens=long_new)
        total = PROMPT_LEN + long_new
        kv_len = total - 1                        # last token's KV not computed
        assert out.shape == (1, total), f"{mode} long-text shape anomaly"
        assert cache.get_seq_length() == kv_len
        layer0 = cache.layers[0]
        assert layer0._key_stream.n_archived > 0, f"{mode} archived nothing"
        assert kv_len - layer0._key_stream.n_archived < 8, \
            "the raw tail must be less than one block"
        raw = kv_len * 1 * 2 * 32 * 4 * 2
        assert cache.storage_bytes() < expect * raw, \
            f"{mode} storage not compressed: {cache.storage_bytes()} vs raw {raw}"


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
