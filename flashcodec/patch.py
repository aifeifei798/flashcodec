"""一键热补丁：让任意 HuggingFace CausalLM 默认走 FlashCodecCache。

机制（零侵入）：标准 HF 模型的 KV 写入/读取统一收敛在 Cache 接口——
注意力层计算出 key/value 后调用 ``cache.update(k, v, layer_idx, ...)``
拿回全历史 KV。因此把 Cache 对象换成 FlashCodecCache（transformers 5.x
原生 Cache 子类），即完成"KV 写入与读取逻辑的替换"，无需针对每种注意力
实现（GQA/MQA/RoPE 变体）做模型手术。

- `apply_flash_codec(model, ...)`：包装 `model.generate` 与 `model.forward`，
  当请求 `use_cache` 且未显式传 `past_key_values` 时自动注入配置好的
  FlashCodecCache；仍会遍历 model.modules() 识别标准 KV 投影注意力模块
  （带 k_proj/v_proj）做校验与报告；
- `remove_flash_codec(model)`：完整还原两个包装。
"""

import functools
import warnings

from .cache import HAS_HF_CACHE, CodecKVCache, FlashCodecCache

__all__ = ["apply_flash_codec", "remove_flash_codec", "undo_flash_codec",
           "find_attention_modules"]


def find_attention_modules(model):
    """识别标准 Self-Attention / KV 投影模块（含 k_proj 与 v_proj）。"""
    return [m for m in model.modules()
            if hasattr(m, "k_proj") and hasattr(m, "v_proj")]


def _make_factory(mode, block_size, kwargs, sink, config=None):
    codec_kw = dict(keep_freqs=kwargs.get("keep_freqs", 2),
                    keep_ratio=kwargs.get("keep_ratio", 1.0))
    if HAS_HF_CACHE:
        def factory():
            c = None
            if config is not None:              # 混合架构（如 Qwen3.5）按
                try:                            # layer_types 只给全注意力层挂 codec
                    c = FlashCodecCache.from_config(
                        config, mode=mode, block_size=block_size, **codec_kw)
                except Exception:
                    c = None
            if c is None:
                c = FlashCodecCache(mode=mode, block_size=block_size,
                                    **codec_kw)
            sink.append(c)
            return c
    else:                                       # 无 transformers 的保底
        def factory():
            c = CodecKVCache(mode=mode, block_size=block_size, **codec_kw)
            sink.append(c)
            return c
    return factory


def apply_flash_codec(model, mode="jpeg", block_size=8, **kwargs):
    """一行热补丁：为 model 注入编解码 KV Cache（transformers 5.x 原生）。

    支持 mode="jpeg"（kwargs: keep_freqs，默认 2）与
    mode="png"（kwargs: keep_ratio，默认 1.0）。
    之后每次 `model.generate(..., use_cache=True)`（未显式传
    past_key_values）都会自动使用 FlashCodecCache。返回 model，
    `remove_flash_codec(model)` 可随时还原。"""
    if getattr(model, "_flashcodec", None) is not None:
        warnings.warn("model 已打过 flashcodec 补丁，先 remove_flash_codec")
        return model

    attns = find_attention_modules(model)
    caches = []                                 # 供测试/诊断检查实际注入的缓存
    factory = _make_factory(mode, block_size, kwargs, caches,
                            config=getattr(model, "config", None))

    base_generate = model.generate
    base_forward = model.forward

    @functools.wraps(base_generate)
    def wrapped_generate(*args, **kw):
        use_cache = kw.get("use_cache")
        if (use_cache is None or use_cache) and kw.get("past_key_values") is None:
            kw["past_key_values"] = factory()
        return base_generate(*args, **kw)

    @functools.wraps(base_forward)
    def wrapped_forward(*args, **kw):
        if kw.get("use_cache") and kw.get("past_key_values") is None:
            kw["past_key_values"] = factory()
        return base_forward(*args, **kw)

    model.generate = wrapped_generate
    model.forward = wrapped_forward
    model._flashcodec = {
        "mode": mode, "caches": caches, "attention_modules": attns,
        "undo": lambda: _restore(model, base_generate, base_forward)}
    if not attns:
        warnings.warn("未找到带 k_proj/v_proj 的注意力模块——若模型使用非标准 "
                      "KV 接口，补丁可能不生效")
    return model


def _restore(model, base_generate, base_forward):
    model.generate = base_generate
    model.forward = base_forward
    model._flashcodec = None


def remove_flash_codec(model):
    """撤销 apply_flash_codec 打的补丁（generate 与 forward 均还原）。"""
    info = getattr(model, "_flashcodec", None)
    if info is not None:
        info["undo"]()


undo_flash_codec = remove_flash_codec          # 旧名兼容
