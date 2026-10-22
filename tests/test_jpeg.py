"""JPEG (frequency-domain lossy) codec tests: orthonormality,
reconstruction fidelity, monotonicity in K, gradients."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from flashcodec import JPEGCodec, dct_basis, dct_compress, dct_decompress

torch.manual_seed(0)

B, H, S, D, R = 2, 4, 64, 32, 8
x = torch.randn(B, H, S, D)


def test_basis_orthonormal():
    m = dct_basis(R, R)
    assert torch.allclose(m @ m.t(), torch.eye(R), atol=1e-6), \
        "basis rows must be orthonormal"


def test_lossless_at_k_equals_r():
    coef = dct_compress(x, R, R)
    rec = dct_decompress(coef, R, out_len=S)
    d = (rec - x).abs().max().item()
    assert d < 1e-5, (
        f"keep=r must be lossless (fp32 GEMM round-trip), "
        f"maxdiff={d:.2e}")


def test_low_freq_fidelity_cosine_gt_0_99():
    smooth = torch.cumsum(torch.randn(B, H, S, D) * 0.1, dim=-2)  # autocorrelated
    rec = dct_decompress(dct_compress(smooth, R, 2), R, out_len=S)
    cos = torch.nn.functional.cosine_similarity(
        rec.flatten(), smooth.flatten(), dim=0).item()
    assert cos > 0.99, f"r=8,k=2 cosine similarity must be >0.99, got {cos:.4f}"


def test_k_monotonicity():
    smooth = torch.cumsum(torch.randn(B, H, S, D) * 0.1, dim=-2)
    e2 = (dct_decompress(dct_compress(smooth, R, 2), R, S) - smooth).abs().mean()
    e4 = (dct_decompress(dct_compress(smooth, R, 4), R, S) - smooth).abs().mean()
    assert e4 < e2, (
        f"error must decrease monotonically with K: "
        f"err2={e2:.4f} err4={e4:.4f}")


def test_padding_and_shapes():
    odd = torch.randn(B, H, 50, D)                     # not a whole block
    rec = dct_decompress(dct_compress(odd, R, 2), R, out_len=50)
    assert rec.shape == odd.shape, "tail padding must be cropped by out_len"
    flat = torch.randn(100, D)                          # any leading batch dims
    assert dct_decompress(dct_compress(flat, R, 2), R, 100).shape == flat.shape


def test_autograd_differentiable():
    xr = torch.randn(B, H, S, D, requires_grad=True)
    rec = dct_decompress(dct_compress(xr, R, 2), R, out_len=S)
    rec.pow(2).mean().backward()
    assert xr.grad is not None and torch.isfinite(xr.grad).all()


def test_codec_class_api():
    c = JPEGCodec(block_size=R, keep_freqs=2)
    coef = c.compress(x)
    rec = c.decompress(coef, out_len=S)
    assert rec.shape == x.shape
    assert abs(c.storage_ratio() - 0.25) < 1e-9       # k/r = 2/8


def test_int8_quantization_quality_and_storage():
    """int8 coefficients + per-(block, freq) scale: near-identical quality to
    the fp32 coefficients and a real 4x smaller archive."""
    from flashcodec import dequantize_coeffs, quantize_coeffs

    smooth = torch.cumsum(torch.randn(B, H, S, D) * 0.1, dim=-2)
    coef = dct_compress(smooth, R, 2)
    q, s = quantize_coeffs(coef)
    assert q.dtype == torch.int8 and q.abs().max().item() <= 127
    assert s.shape == coef.shape[:-1] + (1,)
    deq = dequantize_coeffs(q, s)
    err = (deq - coef).abs().max().item()
    assert err <= s.max().item() / 2 + 1e-6, \
        f"quantization error exceeds half a step: {err:.3e}"

    cos_ref = torch.nn.functional.cosine_similarity(
        dct_decompress(coef, R, S).flatten(), smooth.flatten(), dim=0).item()
    cos_q = torch.nn.functional.cosine_similarity(
        dct_decompress(deq, R, S).flatten(), smooth.flatten(), dim=0).item()
    assert cos_q > 0.99, f"int8 cosine too low: {cos_q:.5f}"
    assert cos_q >= cos_ref - 1e-3, \
        f"int8 quantization must not degrade quality: {cos_ref:.5f} -> {cos_q:.5f}"


def test_codec_int8_api():
    from flashcodec import make_codec

    c = make_codec("jpeg", block_size=R, keep_freqs=2, quant_bits=8)
    assert c.quantized and c.quant_bits == 8
    q, s = c.quantize(c.compress(x))
    rec = c.decompress(q, out_len=S, scales=s)
    assert rec.shape == x.shape and rec.dtype == x.dtype
    assert c.storage_ratio() < 0.25 / 3, "int8 must beat the k/r geometric ratio"

    plain = JPEGCodec(block_size=R, keep_freqs=2)
    assert not plain.quantized and plain.quantize(plain.compress(x)) == (None, None)

    try:
        make_codec("png", quant_bits=8)
        raise AssertionError("png must reject quant_bits (lossless tier)")
    except ValueError:
        pass


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
