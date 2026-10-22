"""Unified codec KV cache manager.

Design:
- Each KV stream (key/value x layer) keeps internally a ``raw_buffer``
  (tail tokens that do not fill a full block, held at full fidelity) and a
  ``compressed_storage`` (compressed archive of completed blocks: jpeg keeps
  low-frequency coefficients / png keeps anchors + deltas);
- Once block_size tokens have accumulated, batched archive compression is
  triggered (a single GEMM, no loops);
- Causal safety: only "complete blocks" are archived, the current partial
  block always stays raw — future perturbations at any point of incremental
  decoding never rewrite already-archived tensors;
- Dual interface: ``forward(key_seq, value_seq)`` tiles the full sequence
  block by block; ``decode_step(new_k, new_v)`` appends a single step and
  archives automatically across block boundaries;
- Duck-typed compatible with HuggingFace ``Cache`` (update / get_seq_length /
  key_cache / value_cache), so it can be injected directly into generate.

Tensor contract: KV layout (batch, num_heads, seq_len, head_dim), compression
axis dim=-2.
"""

import torch

from .codecs import JPEGCodec, PNGCodec, dct_decompress, png_reconstruct

try:  # native transformers 5.x Cache ecosystem (layer container mode)
    from functools import partial

    from transformers.cache_utils import Cache as _HFCache
    from transformers.cache_utils import DynamicLayer as _HFDynamicLayer
    HAS_HF_CACHE = True
except Exception:                          # fallback when transformers is absent
    _HFCache = object
    _HFDynamicLayer = object
    HAS_HF_CACHE = False


def make_codec(mode="jpeg", block_size=8, keep_freqs=2, keep_ratio=1.0):
    """Build a codec instance from a mode name."""
    if mode == "jpeg":
        return JPEGCodec(block_size=block_size, keep_freqs=keep_freqs)
    if mode == "png":
        return PNGCodec(block_size=block_size, keep_ratio=keep_ratio)
    raise ValueError(f"unknown mode: {mode!r} (expected 'jpeg' | 'png')")


class _CodecStream:
    """Single (B,H,S,D) stream: raw tail + capacity-doubling preallocated archive.

    Archive storage uses std::vector-style geometric capacity doubling
    (starting at 16 blocks, doubling when full): each full block is written
    once as an in-place slice (no reallocation), and read() runs a single
    large GEMM / cumsum over the valid region — eliminating per-archive
    Python loops and fragmented kernel launches entirely.
    jpeg: ``_coeffs (B,H,capacity,k,D)`` (input dtype);
    png:  ``_deltas (B,H,capacity,r-1,D)`` (fp32) + ``_anchors (B,H,capacity,D)``.
    """

    _INITIAL_BLOCKS = 16

    def __init__(self, codec):
        self.codec = codec
        self.raw = None          # (B,H,t,D), t < block_size
        self.n_archived = 0      # total number of archived tokens
        self.n_blocks = 0        # blocks used in the archive region
        self.meta = None         # (B, H, D, elem_size)
        self.dtype = None        # stream input dtype (restored; SDPA dtype contract)
        self._capacity = 0       # blocks allocated in the archive (geometric growth)
        self._coeffs = None      # jpeg: (B,H,capacity,k,D)
        self._deltas = None      # png:  (B,H,capacity,r-1,D) fp32
        self._anchors = None     # png:  (B,H,capacity,D)

    @property
    def block_size(self):
        return self.codec.block_size

    def _grow(self, need_blocks, like):
        """Geometric growth: when capacity is insufficient, double until
        >= need_blocks and copy the old data over verbatim."""
        if need_blocks <= self._capacity:
            return
        cap = max(self._INITIAL_BLOCKS, 1)
        while cap < need_blocks:
            cap *= 2
        B, H, D = like.shape[0], like.shape[1], like.shape[-1]
        if self.codec.name == "jpeg":
            new = torch.empty((B, H, cap, self.codec.keep_freqs, D),
                              dtype=self.dtype, device=like.device)
            if self.n_blocks:
                new[:, :, :self.n_blocks].copy_(self._coeffs[:, :, :self.n_blocks])
            self._coeffs = new
        else:
            r = self.block_size
            new_d = torch.empty((B, H, cap, r - 1, D), dtype=torch.float32,
                                device=like.device)
            new_a = torch.empty((B, H, cap, D), dtype=self.dtype,
                                device=like.device)
            if self.n_blocks:
                new_d[:, :, :self.n_blocks].copy_(self._deltas[:, :, :self.n_blocks])
                new_a[:, :, :self.n_blocks].copy_(self._anchors[:, :, :self.n_blocks])
            self._deltas, self._anchors = new_d, new_a
        self._capacity = cap

    def _archive(self, head):
        """Compress full blocks head (..., n_full*r, D) into the archive
        region (in-place slice)."""
        n_full = head.shape[-2] // self.block_size
        self._grow(self.n_blocks + n_full, head)
        sl = slice(self.n_blocks, self.n_blocks + n_full)
        if self.codec.name == "jpeg":
            self._coeffs[:, :, sl].copy_(self.codec.compress(head))
        else:
            delta, anchor = self.codec.filter(head)
            self._deltas[:, :, sl].copy_(delta)
            self._anchors[:, :, sl].copy_(anchor)
        self.n_blocks += n_full

    def append(self, tokens):
        """Append (B,H,m,D); full blocks are batch-compressed into the archive
        immediately (a single batched GEMM)."""
        r = self.block_size
        if self.meta is None:
            self.meta = (tokens.shape[0], tokens.shape[1], tokens.shape[-1],
                         tokens.element_size())
            self.dtype = tokens.dtype        # archive/read restore the input dtype
        buf = tokens if self.raw is None else torch.cat(
            [self.raw, tokens], dim=-2)
        total = buf.shape[-2]
        n_full = total // r
        if n_full > 0:
            self._archive(buf[..., :n_full * r, :])
            self.n_archived += n_full * r
        self.raw = (buf[..., n_full * r:, :].contiguous()
                    if total > n_full * r else None)

    def read(self):
        """Reconstructed view (B,H,S,D): a single decode of the valid archive
        region + concatenation of the raw tail.
        Everything is restored to the stream input dtype (attention q/k/v dtype
        contract)."""
        if self.n_blocks == 0:
            return self.raw
        if self.codec.name == "jpeg":
            rec = dct_decompress(self._coeffs[:, :, :self.n_blocks],
                                 self.block_size)
            if rec.dtype != self.dtype:
                rec = rec.to(self.dtype)
        else:
            rec = png_reconstruct(self._deltas[:, :, :self.n_blocks],
                                  self._anchors[:, :, :self.n_blocks])
        if self.raw is not None:
            rec = torch.cat([rec, self.raw], dim=-2)
        return rec

    def reset(self):
        """Zero the contents but keep the capacity: crop / refill reuse the
        allocated VRAM instead of re-allocating it over and over."""
        self.raw = None
        self.n_archived = 0
        self.n_blocks = 0

    @property
    def seq_len(self):
        return self.n_archived + (0 if self.raw is None else self.raw.shape[-2])

    def storage_bytes(self):
        """Logical storage (valid region + raw tail), excluding the idle
        capacity reserved by the doubling preallocation."""
        total = 0
        if self.n_blocks:
            view = (self._coeffs if self.codec.name == "jpeg"
                    else self._deltas)
            total += view[:, :, :self.n_blocks].numel() * view.element_size()
            if self.codec.name == "png":
                total += (self._anchors[:, :, :self.n_blocks].numel()
                          * self._anchors.element_size())
        if self.raw is not None:
            total += self.raw.numel() * self.raw.element_size()
        return total


class CodecKVCache:
    """Codec KV cache (multi-layer, duck-typed as an HF Cache).

    Usage::

        cache = CodecKVCache(mode="jpeg", block_size=8, keep_freqs=2)
        k, v = cache.update(k_states, v_states, layer_idx=0)   # incremental
        k, v = cache.forward(k_seq, v_seq, layer_idx=0)        # full sequence
    """

    def __init__(self, mode="jpeg", block_size=8, keep_freqs=2,
                 keep_ratio=1.0):
        self.mode = mode
        self.codec = make_codec(mode, block_size, keep_freqs, keep_ratio)
        self._keys = {}
        self._values = {}

    # ---------- internals ----------
    def _stream(self, store, layer_idx):
        st = store.get(layer_idx)
        if st is None:
            st = _CodecStream(self.codec)
            store[layer_idx] = st
        return st

    # ---------- dual interface ----------
    def update(self, key_states, value_states, layer_idx=0):
        """Incremental autoregressive interface: append new (B,H,m,D) and
        archive automatically across block boundaries; returns the layer's
        reconstructed (key_view, value_view)."""
        self._stream(self._keys, layer_idx).append(key_states)
        self._stream(self._values, layer_idx).append(value_states)
        return (self._keys[layer_idx].read(),
                self._values[layer_idx].read())

    decode_step = update

    def forward(self, key_seq, value_seq, layer_idx=0):
        """Full-sequence forward interface: tile the sequence dimension block
        by block directly (reset the layer, then fill it in one shot); returns
        the reconstructed (key_view, value_view)."""
        self.reset(layer_idx)
        return self.update(key_seq, value_seq, layer_idx)

    def reset(self, layer_idx=None):
        if layer_idx is None:
            self._keys.clear()
            self._values.clear()
        else:
            self._keys.pop(layer_idx, None)
            self._values.pop(layer_idx, None)

    # ---------- HF Cache duck typing ----------
    @property
    def key_cache(self):
        mx = max(self._keys, default=-1)
        return [self._keys[i].read() if i in self._keys else None
                for i in range(mx + 1)]

    @property
    def value_cache(self):
        mx = max(self._values, default=-1)
        return [self._values[i].read() if i in self._values else None
                for i in range(mx + 1)]

    def get_seq_length(self, layer_idx=0):
        st = self._keys.get(layer_idx)
        return 0 if st is None else st.seq_len

    def get_max_length(self):
        return None                     # no upper bound

    def get_usable_length(self, new_seq_length, layer_idx=0):
        return self.get_seq_length(layer_idx)

    # ---------- diagnostics ----------
    def storage_bytes(self, layer_idx=None):
        layers = ([layer_idx] if layer_idx is not None
                  else sorted(set(self._keys) | set(self._values)))
        return sum(self._keys[i].storage_bytes() for i in layers if i in self._keys
                   ) + sum(self._values[i].storage_bytes()
                           for i in layers if i in self._values)

    def stats(self, layer_idx=0):
        ks = self._keys.get(layer_idx)
        if ks is None or ks.meta is None:
            return {"tokens": 0, "archived": 0, "raw": 0,
                    "storage_bytes": 0, "raw_bytes": 0}
        B, H, D, elem = ks.meta
        tokens = ks.seq_len
        return {"tokens": tokens, "archived": ks.n_archived,
                "raw": tokens - ks.n_archived,
                "storage_bytes": self.storage_bytes(layer_idx),
                "raw_bytes": tokens * B * H * D * elem * 2}

    def __repr__(self):
        return (f"CodecKVCache(mode={self.mode!r}, codec={self.codec}, "
                f"layers={sorted(set(self._keys) | set(self._values))})")


# ============================================================
# transformers 5.x native Cache backend (layer container mode)
# ============================================================
class FlashCodecLayer(_HFDynamicLayer):
    """transformers 5.x CacheLayer: replaces DynamicLayer's cat-style growth
    with a codec stream (raw tail + archived full blocks). update returns the
    reconstructed, causally visible KV for the attention layer to consume
    directly — the write and read replacements are both contained in this
    layer, requiring zero model surgery."""

    def __init__(self, mode="jpeg", block_size=8, keep_freqs=2,
                 keep_ratio=1.0):
        super().__init__()
        codec = make_codec(mode, block_size, keep_freqs, keep_ratio)
        self.mode = mode
        self.codec = codec
        self._key_stream = _CodecStream(codec)
        self._value_stream = _CodecStream(codec)

    def lazy_initialization(self, key_states, value_states):
        self.dtype, self.device = key_states.dtype, key_states.device
        self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self.values = torch.tensor([], dtype=self.dtype, device=self.device)
        self.is_initialized = True

    def update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        self._key_stream.append(key_states)
        self._value_stream.append(value_states)
        self.keys = self._key_stream.read()
        self.values = self._value_stream.read()
        return self.keys, self.values

    def get_seq_length(self):
        return self._key_stream.seq_len

    def get_max_length(self):
        return -1                            # dynamic cache, no upper bound

    def crop(self, tokens_to_remove):
        """Speculative decoding and similar paths crop the cache: rebuild the
        archive from the cropped reconstructed view (prototype semantics:
        rebuilding filters the already-compressed part once more)."""
        if tokens_to_remove > 0:
            raise ValueError("crop only supports negative values "
                             "(remove trailing tokens)")
        if tokens_to_remove == 0:
            return
        self._key_stream.reset()
        self._value_stream.reset()
        self.keys = self.keys[..., :tokens_to_remove, :]
        self.values = self.values[..., :tokens_to_remove, :]
        if self.keys.shape[-2] > 0:
            self._key_stream.append(self.keys)
            self._value_stream.append(self.values)

    def reset(self):
        self._key_stream.reset()
        self._value_stream.reset()
        super().reset()

    def offload(self):
        pass                     # prototype has no layer offload (archives are off-CPU)

    def prefetch(self):
        pass

    def storage_bytes(self):
        return (self._key_stream.storage_bytes()
                + self._value_stream.storage_bytes())


class FlashCodecCache(_HFCache):
    """transformers 5.x native codec KV cache (first-class backend).

    Usage 1 (explicit)::

        cache = FlashCodecCache(mode="jpeg", block_size=8, keep_freqs=2)
        model.generate(input_ids, past_key_values=cache, use_cache=True)

    Usage 2 (automatic injection via the apply_flash_codec hot patch).
    The Cache base class lazily creates layers through
    layer_class_to_replicate and forwards update to FlashCodecLayer;
    get_seq_length / get_max_length / crop and friends come for free by
    inheritance.
    """

    def __init__(self, mode="jpeg", block_size=8, keep_freqs=2,
                 keep_ratio=1.0):
        if not HAS_HF_CACHE:
            raise ImportError(
                "FlashCodecCache requires transformers 5.x (this project only "
                "supports the installed 5.20 Cache ecosystem); use "
                "CodecKVCache when transformers is unavailable.")
        super().__init__(
            layers=None,
            layer_class_to_replicate=partial(
                FlashCodecLayer, mode=mode, block_size=block_size,
                keep_freqs=keep_freqs, keep_ratio=keep_ratio))
        self.mode = mode
        self.codec_cfg = dict(block_size=block_size, keep_freqs=keep_freqs,
                              keep_ratio=keep_ratio)

    @property
    def _seen_tokens(self):
        """Counter attribute familiar from older transformers versions (unused
        internally by 5.x, kept for compatibility)."""
        return self.get_seq_length()

    @classmethod
    def from_config(cls, config, mode="jpeg", block_size=8, keep_freqs=2,
                    keep_ratio=1.0):
        """Build the mixed-layer list from the model config (Qwen3.5-style
        hybrid architectures): only full_attention layers get a codec layer,
        while the remaining layer types (linear_attention / sliding, etc.)
        keep their native dynamic layer — consistent with how
        DynamicCache(config=...) dispatches, so native interfaces such as
        linear-attention recurrent state stay unaffected."""
        from transformers.cache_utils import (DYNAMIC_LAYER_TYPE_MAPPING,
                                              get_layer_types_and_kwargs)
        decoder = (config.get_text_config(decoder=True)
                   if hasattr(config, "get_text_config") else config)
        try:
            layer_types, per_layer_kwargs = get_layer_types_and_kwargs(decoder)
        except Exception:                        # legacy config, no per_layer_config
            layer_types = list(getattr(decoder, "layer_types",
                                       ["full_attention"] *
                                       getattr(decoder, "num_hidden_layers", 1)))
            per_layer_kwargs = [{} for _ in layer_types]
        layers = []
        for ltype, lkw in zip(layer_types, per_layer_kwargs):
            if ltype == "full_attention":
                layers.append(FlashCodecLayer(
                    mode=mode, block_size=block_size, keep_freqs=keep_freqs,
                    keep_ratio=keep_ratio))
            else:
                layers.append(DYNAMIC_LAYER_TYPE_MAPPING[ltype](**lkw))
        obj = cls.__new__(cls)
        _HFCache.__init__(obj, layers=layers, layer_class_to_replicate=None)
        obj.mode = mode
        obj.codec_cfg = dict(block_size=block_size, keep_freqs=keep_freqs,
                             keep_ratio=keep_ratio)
        return obj

    def codec_layer_indices(self):
        """Indices of the layers that actually carry a codec (i.e. the
        full_attention layers in hybrid architectures)."""
        return [i for i, layer in enumerate(self.layers)
                if isinstance(layer, FlashCodecLayer)]

    def to_legacy_cache(self):
        """Export to the legacy {layer_idx: (keys, values)} format
        (reconstructed full views)."""
        return {i: (layer.keys, layer.values)
                for i, layer in enumerate(self.layers)}

    @classmethod
    def from_legacy_cache(cls, legacy, **cfg):
        """Rebuild from the legacy {(layer_idx): (k, v)} format (the whole
        span goes through the codec once more)."""
        cache = cls(**cfg)
        for layer_idx in sorted(legacy):
            k, v = legacy[layer_idx]
            cache.update(k, v, layer_idx)
        return cache

    def storage_bytes(self, layer_idx=None):
        layers = (self.layers if layer_idx is None
                  else [self.layers[layer_idx]])
        return sum(layer.storage_bytes() for layer in layers
                   if isinstance(layer, FlashCodecLayer))
