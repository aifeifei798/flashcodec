"""统一编解码 KV Cache 管理器。

设计：
- 每条 KV 流（key/value × layer）内部维护
  ``raw_buffer``（未满一个块的尾部 token，保持原始高保真）与
  ``compressed_storage``（完成块的压缩归档：jpeg 存低频系数 / png 存
  锚点+差分）；
- 一旦累积满 block_size 个 token，触发批量归档压缩（一次 GEMM，无循环）；
- 因果安全：只归档"完整块"，当前未满块永远是 raw——增量解码时任意时刻
  的未来扰动都不会改写已归档张量；
- 双接口：``forward(key_seq, value_seq)`` 全量分块平铺；
  ``decode_step(new_k, new_v)`` 单步追加、跨块边界自动归档；
- 鸭子类型兼容 HuggingFace ``Cache``（update / get_seq_length /
  key_cache / value_cache），可直接注入 generate 流程。

张量契约：KV 布局 (batch, num_heads, seq_len, head_dim)，压缩轴 dim=-2。
"""

import torch

from .codecs import JPEGCodec, PNGCodec, dct_decompress, png_reconstruct

try:  # transformers 5.x 原生 Cache 生态（layer 容器模式）
    from functools import partial

    from transformers.cache_utils import Cache as _HFCache
    from transformers.cache_utils import DynamicLayer as _HFDynamicLayer
    HAS_HF_CACHE = True
except Exception:                                    # 未安装 transformers 时保底
    _HFCache = object
    _HFDynamicLayer = object
    HAS_HF_CACHE = False


def make_codec(mode="jpeg", block_size=8, keep_freqs=2, keep_ratio=1.0):
    """按模式名构造编解码器实例。"""
    if mode == "jpeg":
        return JPEGCodec(block_size=block_size, keep_freqs=keep_freqs)
    if mode == "png":
        return PNGCodec(block_size=block_size, keep_ratio=keep_ratio)
    raise ValueError(f"未知 mode: {mode!r}（可选 'jpeg' | 'png'）")


class _CodecStream:
    """单条 (B,H,S,D) 流：raw 尾巴 + 倍增预分配的压缩归档。

    归档存储采用 std::vector 式几何倍增预分配（初始 16 块，装满翻倍）：
    每满块只做一次原地切片写入（无重新分配），read() 对有效区做单次
    大 GEMM / cumsum——彻底消灭逐归档 Python 循环与碎片 kernel 发射。
    jpeg: ``_coeffs (B,H,capacity,k,D)``（输入 dtype）；
    png:  ``_deltas (B,H,capacity,r-1,D)``（fp32）+ ``_anchors (B,H,capacity,D)``。
    """

    _INITIAL_BLOCKS = 16

    def __init__(self, codec):
        self.codec = codec
        self.raw = None          # (B,H,t,D)，t < block_size
        self.n_archived = 0      # 已归档 token 总数
        self.n_blocks = 0        # 归档区已用块数
        self.meta = None         # (B, H, D, elem_size)
        self.dtype = None        # 流输入 dtype（读取统一还原，SDPA dtype 契约）
        self._capacity = 0       # 归档区已分配块数（几何倍增）
        self._coeffs = None      # jpeg: (B,H,capacity,k,D)
        self._deltas = None      # png:  (B,H,capacity,r-1,D) fp32
        self._anchors = None     # png:  (B,H,capacity,D)

    @property
    def block_size(self):
        return self.codec.block_size

    def _grow(self, need_blocks, like):
        """几何倍增扩容：容量不足时翻倍到 ≥ need_blocks，旧数据原样拷入。"""
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
        """把满块 head (..., n_full*r, D) 压缩写入归档区（原地切片）。"""
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
        """追加 (B,H,m,D)；满块即批量压缩写入归档区（一次批量 GEMM）。"""
        r = self.block_size
        if self.meta is None:
            self.meta = (tokens.shape[0], tokens.shape[1], tokens.shape[-1],
                         tokens.element_size())
            self.dtype = tokens.dtype        # 归档/读取统一回输入 dtype
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
        """重构视图 (B,H,S,D)：有效归档区单次解码 + raw 尾巴拼接。
        统一还原到流输入 dtype（注意力 q/k/v 的 dtype 契约）。"""
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
        """清零内容但保留容量：crop / 重填复用已分配显存，不再反复申请。"""
        self.raw = None
        self.n_archived = 0
        self.n_blocks = 0

    @property
    def seq_len(self):
        return self.n_archived + (0 if self.raw is None else self.raw.shape[-2])

    def storage_bytes(self):
        """逻辑存储（有效区 + raw 尾），不含倍增预留的空闲容量。"""
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
    """编解码 KV Cache（多 layer，HF Cache 鸭子类型）。

    用法::

        cache = CodecKVCache(mode="jpeg", block_size=8, keep_freqs=2)
        k, v = cache.update(k_states, v_states, layer_idx=0)   # 增量
        k, v = cache.forward(k_seq, v_seq, layer_idx=0)        # 全量
    """

    def __init__(self, mode="jpeg", block_size=8, keep_freqs=2,
                 keep_ratio=1.0):
        self.mode = mode
        self.codec = make_codec(mode, block_size, keep_freqs, keep_ratio)
        self._keys = {}
        self._values = {}

    # ---------- 内部 ----------
    def _stream(self, store, layer_idx):
        st = store.get(layer_idx)
        if st is None:
            st = _CodecStream(self.codec)
            store[layer_idx] = st
        return st

    # ---------- 双接口 ----------
    def update(self, key_states, value_states, layer_idx=0):
        """增量自回归接口：追加新 (B,H,m,D)，跨块边界自动压缩归档；
        返回该层重构后的 (key_view, value_view)。"""
        self._stream(self._keys, layer_idx).append(key_states)
        self._stream(self._values, layer_idx).append(value_states)
        return (self._keys[layer_idx].read(),
                self._values[layer_idx].read())

    decode_step = update

    def forward(self, key_seq, value_seq, layer_idx=0):
        """全量前向接口：在序列维度直接分块平铺（重置该层后一次性填充），
        返回重构后的 (key_view, value_view)。"""
        self.reset(layer_idx)
        return self.update(key_seq, value_seq, layer_idx)

    def reset(self, layer_idx=None):
        if layer_idx is None:
            self._keys.clear()
            self._values.clear()
        else:
            self._keys.pop(layer_idx, None)
            self._values.pop(layer_idx, None)

    # ---------- HF Cache 鸭子类型 ----------
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
        return None                     # 无上限

    def get_usable_length(self, new_seq_length, layer_idx=0):
        return self.get_seq_length(layer_idx)

    # ---------- 诊断 ----------
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
# transformers 5.x 原生 Cache 后端（layer 容器模式）
# ============================================================
class FlashCodecLayer(_HFDynamicLayer):
    """transformers 5.x CacheLayer：把 DynamicLayer 的 cat 式增长替换为
    编解码流（raw 尾 + 满块归档）。update 返回重构后的完整因果可见 KV，
    供注意力层直接使用——写入与读取的替换收敛在这一层，模型零手术。"""

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
        return -1                            # 动态缓存无上限

    def crop(self, tokens_to_remove):
        """投机解码等路径会裁剪缓存：从裁剪后的重构视图重建归档
 （原型语义：重建会把已压缩部分重新过滤一遍）。"""
        if tokens_to_remove > 0:
            raise ValueError("crop 只支持负值（移除末尾 token）")
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
        pass                               # 原型不支持层卸载（流式归档在 CPU 外）

    def prefetch(self):
        pass

    def storage_bytes(self):
        return (self._key_stream.storage_bytes()
                + self._value_stream.storage_bytes())


class FlashCodecCache(_HFCache):
    """transformers 5.x 原生编解码 KV Cache（第一等公民后端）。

    用法一（显式传参）::

        cache = FlashCodecCache(mode="jpeg", block_size=8, keep_freqs=2)
        model.generate(input_ids, past_key_values=cache, use_cache=True)

    用法二（apply_flash_codec 热补丁自动注入）。
    基类 Cache 以 layer_class_to_replicate 懒建层，update 直接下发到
    FlashCodecLayer；get_seq_length / get_max_length / crop 等继承即用。
    """

    def __init__(self, mode="jpeg", block_size=8, keep_freqs=2,
                 keep_ratio=1.0):
        if not HAS_HF_CACHE:
            raise ImportError(
                "FlashCodecCache 需要 transformers 5.x（本项目仅适配当前安装的 "
                "5.20 Cache 生态）；无 transformers 时请使用 CodecKVCache。")
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
        """旧版 transformers 习惯的计数器属性（5.x 内部已不用，保留兼容）。"""
        return self.get_seq_length()

    @classmethod
    def from_config(cls, config, mode="jpeg", block_size=8, keep_freqs=2,
                    keep_ratio=1.0):
        """按模型 config 构建混合层列表（Qwen3.5 等混合架构）：
        只有 full_attention 层挂 codec 层，其余层类型（linear_attention/
        sliding 等）保留原生动态层——与 DynamicCache(config=...) 的分发
        一致，线性注意力的递归状态等原生接口不受影响。"""
        from transformers.cache_utils import (DYNAMIC_LAYER_TYPE_MAPPING,
                                              get_layer_types_and_kwargs)
        decoder = (config.get_text_config(decoder=True)
                   if hasattr(config, "get_text_config") else config)
        try:
            layer_types, per_layer_kwargs = get_layer_types_and_kwargs(decoder)
        except Exception:                        # 老式 config 无 per_layer_config
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
        """真正挂了编解码的层下标（混合架构中即 full_attention 层）。"""
        return [i for i, layer in enumerate(self.layers)
                if isinstance(layer, FlashCodecLayer)]

    def to_legacy_cache(self):
        """导出为 {layer_idx: (keys, values)} 旧格式（重构后的完整视图）。"""
        return {i: (layer.keys, layer.values)
                for i, layer in enumerate(self.layers)}

    @classmethod
    def from_legacy_cache(cls, legacy, **cfg):
        """从旧格式 {(layer_idx): (k, v)} 重建（整段重走过一遍编解码）。"""
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
