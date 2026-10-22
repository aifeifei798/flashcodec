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

from flashcodec import (FlashCodecCache, apply_flash_codec, codec_attention_stats,
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


def test_e_beam_search_lossless_exact():
    """Beam search exercises batch expansion + reorder_cache (archives must be
    gathered, not just the materialized view): the PNG lossless tier still
    matches the unpatched model token for token."""
    base = build_tiny_llama()
    ref = gen(base, num_beams=3)
    apply_flash_codec(base, mode="png", block_size=8, keep_ratio=1.0)
    out = gen(base, num_beams=3)
    assert torch.equal(ref, out), (
        f"beam search diverged on the lossless tier: ref={ref[0, -NEW:].tolist()} "
        f"got={out[0, -NEW:].tolist()}")


def test_f_crop_matches_unpatched():
    """crop (speculative decoding) keeps the exact surviving prefix and a
    correct boundary: after cropping 5 tokens the next-token logits agree with
    the unpatched model fed the same truncated prefix."""
    base = build_tiny_llama()
    ref_model = build_tiny_llama()               # same seed => same weights
    apply_flash_codec(base, mode="png", block_size=8, keep_ratio=1.0)
    cache = FlashCodecCache(mode="png", block_size=8, keep_ratio=1.0)
    with torch.no_grad():
        out = base.generate(input_ids=PROMPT(), do_sample=False,
                            pad_token_id=0, max_new_tokens=NEW,
                            past_key_values=cache)
    total = out.shape[1]
    assert cache.get_seq_length() == total - 1
    before = [layer.keys.clone() for layer in cache.layers]
    cache.crop(-5)
    assert cache.get_seq_length() == total - 6
    for li, layer in enumerate(cache.layers):
        assert torch.equal(layer.keys, before[li][..., :total - 6, :]), \
            f"layer {li}: crop rewrote the surviving prefix"

    # continue one step from the cropped cache vs the unpatched model
    next_tok = out[:, total - 6:total - 5]
    with torch.no_grad():
        ours = base(input_ids=next_tok, past_key_values=cache,
                    use_cache=True).logits[:, -1, :]
        ref = ref_model(input_ids=out[:, :total - 5]).logits[:, -1, :]
    d = (ours - ref).abs().max().item()
    assert torch.equal(ours.argmax(-1), ref.argmax(-1)), \
        f"crop boundary changed the next token (maxdiff={d:.2e})"
    assert d < 1e-2, f"crop boundary logits drifted: maxdiff={d:.2e}"


def test_g_int8_generation():
    """int8 coefficients work end to end through generate and shrink the
    archive well below the same cache in fp32 (the raw tail is shared, so
    compare against the unquantized codec rather than raw KV)."""
    model = build_tiny_llama()
    plain = FlashCodecCache(mode="jpeg", block_size=8, keep_freqs=2)
    gen(model, past_key_values=plain)
    model = build_tiny_llama()
    cache = FlashCodecCache(mode="jpeg", block_size=8, keep_freqs=2,
                            quant_bits=8)
    out = gen(model, past_key_values=cache)
    assert out.shape == (1, PROMPT_LEN + NEW)
    assert cache.get_seq_length() == PROMPT_LEN + NEW - 1
    assert cache.storage_bytes() < 0.7 * plain.storage_bytes(), (
        f"int8 storage not active: {cache.storage_bytes()} vs fp "
        f"{plain.storage_bytes()}")


def test_h_per_layer_schedule():
    """from_config applies a calibrated per-layer budget (mixed keep_freqs /
    quant_bits) and still generates."""
    model = build_tiny_llama()
    cache = FlashCodecCache.from_config(
        model.config, mode="jpeg", block_size=8, keep_freqs=8,
        per_layer={1: dict(keep_freqs=2, quant_bits=8)})
    layers = cache.codec_layer_indices()
    assert len(layers) == model.config.num_hidden_layers
    assert cache.layers[0].codec.keep_freqs == 8
    assert cache.layers[0].codec.quant_bits == 0
    assert cache.layers[1].codec.keep_freqs == 2
    assert cache.layers[1].codec.quant_bits == 8
    out = gen(model, past_key_values=cache)
    assert out.shape == (1, PROMPT_LEN + NEW)
    assert cache.get_seq_length() == PROMPT_LEN + NEW - 1


def test_i_materialization_free_generate():
    """The coefficient-domain attention patch produces the same tokens on the
    lossless tier, wraps every attention module, drops the prefill view, and is
    fully removable."""
    base = build_tiny_llama()
    ref = gen(base)
    model = build_tiny_llama()
    apply_flash_codec(model, mode="png", block_size=8, keep_ratio=1.0,
                      materialization_free=True)
    n_layers = model.config.num_hidden_layers
    assert model._flashcodec["codec_attention_modules"] == n_layers
    assert codec_attention_stats(model) == n_layers
    out = gen(model)
    assert torch.equal(ref, out), "materialization-free lossless run diverged"

    cache = model._flashcodec["caches"][-1]
    assert cache.get_seq_length() == PROMPT_LEN + NEW - 1
    for layer in cache.layers:
        assert layer._key_stream._rec is None, \
            "decode must not keep the reconstructed view"

    remove_flash_codec(model)
    assert codec_attention_stats(model) == 0
    assert gen(model).shape == ref.shape


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
