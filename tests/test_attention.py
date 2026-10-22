"""Materialization-free decode attention tests.

1) The coefficient-domain path matches materialized attention for jpeg
   (plain / int8 / lossless) and png, including GQA head grouping
2) It never builds the reconstructed view (``_rec`` stays None) and does not
   disturb the stream
3) Empty cache and prefill-shaped queries are handled correctly
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from flashcodec import CodecKVCache, codec_decode_attention

torch.manual_seed(0)

B, Hk, S, D = 2, 4, 96, 32
k_seq = torch.cumsum(torch.randn(B, Hk, S, D) * 0.1, dim=-2)
v_seq = torch.cumsum(torch.randn(B, Hk, S, D) * 0.1, dim=-2)


def materialized(q, cache, g):
    kr = cache.key_cache[0].repeat_interleave(g, dim=1)
    vr = cache.value_cache[0].repeat_interleave(g, dim=1)
    p = torch.softmax(q @ kr.transpose(-2, -1) / math.sqrt(D), dim=-1)
    return p @ vr


MODES = [
    ("jpeg", dict(keep_freqs=2)),
    ("jpeg", dict(keep_freqs=8)),                 # lossless DCT basis
    ("jpeg", dict(keep_freqs=2, quant_bits=8)),
    ("png", dict(keep_ratio=1.0)),
    ("png", dict(keep_ratio=0.5)),
]


def test_matches_materialized_attention():
    for g in (1, 2):
        q = torch.randn(B, Hk * g, 1, D)
        for mode, kw in MODES:
            cache = CodecKVCache(mode=mode, block_size=8, **kw)
            cache.forward(k_seq, v_seq)
            out = codec_decode_attention(q, cache._keys[0], cache._values[0])
            ref = materialized(q, cache, g)
            d = (out - ref).abs().max().item()
            assert out.shape == ref.shape
            assert d < 1e-5, f"{mode} {kw} g={g}: maxdiff={d:.2e}"


def test_no_materialization_and_stream_untouched():
    from flashcodec.cache import _CodecStream, make_codec

    q = torch.randn(B, Hk, 1, D)
    cfg = dict(mode="jpeg", block_size=8, keep_freqs=2, quant_bits=8)
    ks, vs = _CodecStream(make_codec(**cfg)), _CodecStream(make_codec(**cfg))
    ks.append(k_seq)
    vs.append(v_seq)
    assert ks._rec is None and vs._rec is None
    n_before, len_before = ks.n_blocks, ks.seq_len
    out = codec_decode_attention(q, ks, vs)
    assert ks._rec is None and vs._rec is None, \
        "the coefficient path must not build the reconstructed view"
    assert ks.n_blocks == n_before and ks.seq_len == len_before
    assert torch.isfinite(out).all()


def test_empty_cache_and_prefill_guard():
    from flashcodec.cache import _CodecStream, make_codec

    cfg = dict(mode="jpeg", block_size=8, keep_freqs=2)
    ks, vs = _CodecStream(make_codec(**cfg)), _CodecStream(make_codec(**cfg))
    out = codec_decode_attention(torch.randn(B, Hk, 1, D), ks, vs)
    assert out.shape == (B, Hk, 1, D) and torch.equal(out, torch.zeros_like(out))

    cache = CodecKVCache(**cfg)
    cache.forward(k_seq, v_seq)
    try:
        codec_decode_attention(torch.randn(B, Hk, 4, D), cache._keys[0],
                               cache._values[0])
        raise AssertionError("prefill-shaped queries must be rejected")
    except ValueError:
        pass


def test_triton_matches_torch_path():
    """The Triton split-K kernel must agree with the torch coefficient path
    (and therefore with the materialized path). CPU-only environments skip."""
    if not torch.cuda.is_available():
        print("   (skipped: requires CUDA)")
        return
    from flashcodec.cache import _CodecStream, make_codec

    torch.manual_seed(0)
    B, Hk, D, S = 1, 8, 128, 4096
    g = 2
    k = torch.randn(B, Hk, S, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, Hk, S, D, device="cuda", dtype=torch.bfloat16)
    q = torch.randn(B, Hk * g, 1, D, device="cuda", dtype=torch.bfloat16)
    for cfg in (dict(keep_freqs=2, quant_bits=8), dict(keep_freqs=4)):
        ks = _CodecStream(make_codec("jpeg", block_size=8, **cfg))
        vs = _CodecStream(make_codec("jpeg", block_size=8, **cfg))
        ks.append(k)
        vs.append(v)
        ref = codec_decode_attention(q, ks, vs, impl="torch")
        tri = codec_decode_attention(q, ks, vs, impl="triton")
        d = (ref.float() - tri.float()).abs().max().item()
        assert d < 2e-3, f"triton vs torch mismatch: {cfg} maxdiff={d:.2e}"
        auto = codec_decode_attention(q, ks, vs)
        assert auto.dtype == q.dtype


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
    print("\nALL PASS" if ok else "\nFAILURES")
    sys.exit(0 if ok else 1)
