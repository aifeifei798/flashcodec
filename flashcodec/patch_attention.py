"""Route decode steps through the coefficient-domain attention kernel.

`apply_flash_codec` swaps the cache, but transformers still computes attention
on the reconstructed KV view. This patch replaces the forward of *standard*
Llama/Qwen-style attention modules (q_proj / k_proj / v_proj / o_proj, optional
q_norm / k_norm) so that, for a single-token decode step on a codec layer, the
projection + RoPE part runs as usual and the attention itself is computed by
`codec_decode_attention` — the (S, D) reconstructed view is never built.

Scope / guards (anything else falls back to the original forward):
- single-token query, batch size 1, eval mode
- the cache layer must be a ``FlashCodecLayer`` (i.e. a full-attention layer)
- no sliding-window attention
- ``position_embeddings`` must be provided (5.x standard signature)

The fallback is silent by design (mixed models have linear-attention layers that
never hit this path); `codec_attention_stats` reports how many layers are wrapped.
"""

import importlib

import torch

from .attention import codec_decode_attention
from .cache import FlashCodecLayer

_ROPE_CACHE = {}


def _rope_fn(module):
    """The `apply_rotary_pos_emb` of the model file that defines this module."""
    mod_name = type(module).__module__
    if mod_name not in _ROPE_CACHE:
        mod = importlib.import_module(mod_name)
        _ROPE_CACHE[mod_name] = getattr(mod, "apply_rotary_pos_emb", False)
    return _ROPE_CACHE[mod_name] or None


def _attention_forward(module, orig, hidden_states,
                       position_embeddings=None, attention_mask=None,
                       past_key_values=None, **kwargs):
    layer = None
    if past_key_values is not None and hasattr(past_key_values, "layers"):
        try:
            layer = past_key_values.layers[module.layer_idx]
        except (IndexError, TypeError):
            layer = None

    usable = (
        isinstance(layer, FlashCodecLayer)
        and not module.training
        and hidden_states.shape[0] == 1          # batch 1: no padding mask
        and hidden_states.shape[-2] == 1         # decode step only
        and position_embeddings is not None
        and getattr(module, "sliding_window", None) is None
        and _rope_fn(module) is not None
    )
    if not usable:
        return orig(hidden_states, position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values, **kwargs)

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, module.head_dim)
    o_in = module.o_proj.in_features
    gated = module.q_proj.out_features == 2 * o_in      # Qwen3.5-style Q gate
    q_raw = module.q_proj(hidden_states)
    gate = None
    if gated:
        q, gate = torch.chunk(
            q_raw.view(*input_shape, -1, module.head_dim * 2), 2, dim=-1)
        gate = gate.reshape(*input_shape, -1)
    else:
        q = q_raw
    q = q.view(hidden_shape)
    k = module.k_proj(hidden_states).view(hidden_shape)
    v = module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    # Qwen3-style QK-norm is applied on (heads, head_dim) before the transpose
    if hasattr(module, "q_norm"):
        q = module.q_norm(q)
    if hasattr(module, "k_norm"):
        k = module.k_norm(k)
    q, k = q.transpose(1, 2), k.transpose(1, 2)

    cos, sin = position_embeddings
    q, k = _rope_fn(module)(q, k, cos, sin)

    if not layer.is_initialized:
        layer.lazy_initialization(k, v)
    # prefill left a full reconstructed view behind; decode does not need it
    for st in (layer._key_stream, layer._value_stream):
        if st._rec is not None:
            st.discard_view()
    layer.keys = layer.values = None            # drop the prefill view reference
    layer._key_stream.append(k)                  # append only: no read()
    layer._value_stream.append(v)
    out = codec_decode_attention(q, layer._key_stream, layer._value_stream,
                                 scale=module.scaling)
    out = out.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    if gated:
        out = out * torch.sigmoid(gate)
    return module.o_proj(out), None


def apply_codec_attention(model):
    """Patch every supported attention module so decode steps use the
    coefficient-domain kernel. Idempotent; returns the number of wrapped
    modules."""
    wrapped = 0
    for module in model.modules():
        if not (hasattr(module, "q_proj") and hasattr(module, "k_proj")
                and hasattr(module, "v_proj") and hasattr(module, "o_proj")
                and hasattr(module, "layer_idx") and hasattr(module, "head_dim")):
            continue
        # only the standard layout: head_dim divides the output width, and the
        # query projection is either plain (Llama/Qwen3) or gated (Qwen3.5)
        o_in = module.o_proj.in_features
        if module.head_dim <= 0 or o_in % module.head_dim:
            continue
        if module.q_proj.out_features not in (o_in, 2 * o_in):
            continue
        if getattr(module, "_flashcodec_orig_forward", None) is not None:
            wrapped += 1
            continue
        orig = module.forward

        def forward(*args, _module=module, _orig=orig, **kw):
            return _attention_forward(_module, _orig, *args, **kw)

        module._flashcodec_orig_forward = orig
        module.forward = forward
        wrapped += 1
    return wrapped


def remove_codec_attention(model):
    """Restore the original forwards. Returns the number restored."""
    restored = 0
    for module in model.modules():
        orig = getattr(module, "_flashcodec_orig_forward", None)
        if orig is not None:
            module.forward = orig
            del module._flashcodec_orig_forward
            restored += 1
    return restored


def codec_attention_stats(model):
    """Number of attention modules currently wrapped."""
    return sum(1 for m in model.modules()
               if getattr(m, "_flashcodec_orig_forward", None) is not None)
