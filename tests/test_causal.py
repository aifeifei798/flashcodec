"""Causal safety and cache-consistency tests:
1) Zero leakage from future perturbations (archived block tensors stay
   bitwise-identical, in both full and incremental modes)
2) Incremental decoding (lossless png) aligns with the full forward
   pass in attention output to within 1e-4
3) The cache view from forward filling is bitwise-identical to the one
   from step-by-step filling
4) Attention-output quality of the lossy jpeg tier (cosine report)
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from flashcodec import CodecKVCache

torch.manual_seed(0)

B, H, S, D, R = 2, 4, 96, 32, 8
k_seq = torch.randn(B, H, S, D)
v_seq = torch.randn(B, H, S, D)


def toy_attn(q, k, v):
    """Single-segment causal attention (B,H,S_q,D): scaled dot product of
    q against the full KV, plus a causal mask."""
    Sq, Sk = q.shape[-2], k.shape[-2]
    logits = q @ k.transpose(-2, -1) / math.sqrt(D)
    mask = torch.ones(Sq, Sk, dtype=torch.bool, device=q.device).tril(
        diagonal=Sk - Sq)
    return torch.softmax(logits.masked_fill(~mask, float("-inf")),
                         dim=-1) @ v


def _fill_incr(mode, **kw):
    cache = CodecKVCache(mode=mode, **kw)
    for t in range(S):
        cache.decode_step(k_seq[..., t:t + 1, :], v_seq[..., t:t + 1, :])
    return cache


def _archive_tensors(st):
    """Snapshot of the archive's valid region (logical view of the
    doubling preallocated storage)."""
    if st.codec.name == "jpeg":
        return [st._coeffs[:, :, :st.n_blocks]]
    return [st._deltas[:, :, :st.n_blocks], st._anchors[:, :, :st.n_blocks]]


def test_future_perturbation_zero_leak_decode():
    c = CodecKVCache(mode="png", block_size=R, keep_ratio=0.5)
    cut = 2 * R                                  # 2 whole blocks archived
    for t in range(cut):
        c.decode_step(k_seq[..., t:t + 1, :], v_seq[..., t:t + 1, :])
    snap = [[t.clone() for t in _archive_tensors(st)]
            for st in (c._keys[0], c._values[0])]
    view_before = c.key_cache[0].clone()
    pert = torch.randn_like(k_seq[..., cut:, :])
    for t in range(cut, S):                      # perturb future tokens
        c.decode_step(pert[..., t - cut:t - cut + 1, :],
                      pert[..., t - cut:t - cut + 1, :])
    for st, snap_i in zip((c._keys[0], c._values[0]), snap):
        for ar, ar0 in zip(_archive_tensors(st), snap_i):
            # The archive is one contiguous doubling-preallocated buffer:
            # its valid region only grows and is never rewritten, so the
            # snapshotted historical prefix must stay bitwise-identical
            n0 = ar0.shape[2]
            assert torch.equal(ar[:, :, :n0], ar0), \
                "a future perturbation rewrote archived tensors"
    assert torch.equal(c.key_cache[0][..., :cut, :], view_before[..., :cut, :]), \
        "a future perturbation rewrote the historical view"


def test_future_perturbation_zero_leak_forward():
    c1 = CodecKVCache(mode="jpeg", block_size=R, keep_freqs=2)
    c2 = CodecKVCache(mode="jpeg", block_size=R, keep_freqs=2)
    cut = 3 * R                                  # block-aligned cut point
    k2 = k_seq.clone()
    k2[..., cut:, :] = torch.randn_like(k2[..., cut:, :])
    c1.forward(k_seq, v_seq)
    c2.forward(k2, v_seq)
    # In full mode every complete block is archived in one pass into the
    # doubling storage (B,H,n_blocks,k,D); the block dim is -3
    s1, s2 = c1._keys[0], c2._keys[0]
    a1, a2 = _archive_tensors(s1)[0], _archive_tensors(s2)[0]
    assert torch.equal(a1[..., :3, :, :], a2[..., :3, :, :]), \
        "in full mode a future perturbation affected the historical archive"
    assert torch.equal(c1.key_cache[0][..., :cut, :],
                       c2.key_cache[0][..., :cut, :])


def test_incremental_vs_full_attention_lossless():
    """With a lossless png cache, step-by-step decoding attention ==
    full forward pass attention (<=1e-4)."""
    q = torch.randn(B, H, S, D)
    c_full = CodecKVCache(mode="png", block_size=R, keep_ratio=1.0)
    kf, vf = c_full.forward(k_seq, v_seq)
    out_full = toy_attn(q, kf, vf)

    c = CodecKVCache(mode="png", block_size=R, keep_ratio=1.0)
    outs = []
    for t in range(S):
        kt, vt = c.decode_step(k_seq[..., t:t + 1, :],
                               v_seq[..., t:t + 1, :])
        outs.append(toy_attn(q[..., t:t + 1, :], kt, vt))
    out_incr = torch.cat(outs, dim=-2)
    d = (out_full - out_incr).abs().max().item()
    assert d < 1e-4, \
        f"incremental decoding disagrees with full forward: maxdiff={d:.2e}"


def test_forward_fill_vs_incremental_view_bitwise():
    c1 = CodecKVCache(mode="png", block_size=R, keep_ratio=1.0)
    k1, v1 = c1.forward(k_seq, v_seq)
    c2 = _fill_incr("png", block_size=R, keep_ratio=1.0)
    assert torch.equal(k1, c2.key_cache[0]) and torch.equal(v1, c2.value_cache[0])
    c3 = CodecKVCache(mode="jpeg", block_size=R, keep_freqs=2)
    k3, _ = c3.forward(k_seq, v_seq)
    c4 = _fill_incr("jpeg", block_size=R, keep_freqs=2)
    assert torch.equal(k3, c4.key_cache[0]), \
        "jpeg full/incremental views must be bitwise-identical"


def test_jpeg_attention_quality_report():
    smooth = torch.cumsum(torch.randn(B, H, S, D) * 0.1, dim=-2)
    q = torch.randn(B, H, S, D)
    out_exact = toy_attn(q, smooth, smooth)
    c = CodecKVCache(mode="jpeg", block_size=R, keep_freqs=2)
    kq, vq = c.forward(smooth, smooth)
    out_q = toy_attn(q, kq, vq)
    cos = torch.nn.functional.cosine_similarity(
        out_exact.flatten(), out_q.flatten(), dim=0).item()
    print(f"   jpeg(r=8,k=2) attention output cosine = {cos:.4f}")
    assert cos > 0.99, f"jpeg tier attention quality too low: cos={cos:.4f}"


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
