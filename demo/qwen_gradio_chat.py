"""Qwen3.5-0.8B × flashcodec 左右对比 Gradio Chat。

用法：python demo/qwen_gradio_chat.py   →   http://127.0.0.1:7860
环境变量：FLASHCODEC_MODEL（默认 Qwen/Qwen3.5-0.8B）、WEBCHAT_HOST、
WEBCHAT_PORT、WEBCHAT_NEW_TOKENS（默认 128）。

布局：左侧原版 baseline，右侧 flashcodec（同一 prompt 顺序生成，逐 token
流式）；底部柱状图对比「节约了什么」——KV Cache 存储与生成耗时相对原版的
变化，外加 token 一致率与绝对数字表。
"""

import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
# 本机模型离线加载，不走代理；socks 代理 URL 会让 httpx/gradio 启动报错
for _k in ("http_proxy", "https_proxy", "all_proxy",
           "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
    os.environ.pop(_k, None)

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          TextIteratorStreamer)
from transformers.cache_utils import get_layer_types_and_kwargs

from flashcodec import apply_flash_codec, remove_flash_codec
from flashcodec.cache import FlashCodecLayer

MODEL = os.environ.get("FLASHCODEC_MODEL", "Qwen/Qwen3.5-0.8B")
HOST = os.environ.get("WEBCHAT_HOST", "127.0.0.1")
PORT = int(os.environ.get("WEBCHAT_PORT", "7860"))
MAX_NEW = int(os.environ.get("WEBCHAT_NEW_TOKENS", "128"))

MODES = {
    "png keep=1.0（无损，逐位一致）": dict(mode="png", block_size=8, keep_ratio=1.0),
    "jpeg k=1（8× 有损）":            dict(mode="jpeg", block_size=8, keep_freqs=1),
    "jpeg k=2（4× 有损）":            dict(mode="jpeg", block_size=8, keep_freqs=2),
    "jpeg k=4（2× 有损）":            dict(mode="jpeg", block_size=8, keep_freqs=4),
}
DEFAULT_MODE = "jpeg k=2（4× 有损）"

_CJK = ["Noto Sans SC", "Source Han Sans CN", "Noto Sans CJK SC",
        "WenQuanYi Micro Hei", "SimHei", "PingFang SC", "Microsoft YaHei"]
matplotlib.rcParams["font.sans-serif"] = _CJK + matplotlib.rcParams["font.sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"加载 {MODEL}（{device}）...")
tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
_dtype = torch.bfloat16 if device == "cuda" else torch.float32
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, dtype=_dtype)
except Exception:
    from transformers import AutoModelForImageTextToText
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, local_files_only=True, dtype=_dtype)
model = model.to(device).eval()
print("就绪 ✅")

gpu_lock = threading.Lock()

# baseline 的显式 DynamicCache 在混合架构（linear_attention）下不可用，
# 改为按 config 解析 full-attention 层的 raw KV 字节——flashcodec 只压缩
# 这些层，两侧 apples-to-apples。
_dec = model.config.get_text_config(decoder=True)
_layer_types, _ = get_layer_types_and_kwargs(_dec)
N_FULL = sum(1 for t in _layer_types if t == "full_attention")
_HKV = _dec.num_key_value_heads
_HEAD_DIM = getattr(_dec, "head_dim", _dec.hidden_size // _dec.num_attention_heads)
_ELEM = torch.empty((), dtype=model.dtype).element_size()


def _baseline_attn_bytes(kv_len):
    """原版 full-attention 层的 raw KV 总字节（K+V）。"""
    return N_FULL * 2 * kv_len * _HKV * _HEAD_DIM * _ELEM


class _IdStreamer(TextIteratorStreamer):
    """记录生成的新 token id（跳过 prompt 首帧），用于一致率统计。"""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.ids = []

    def put(self, value):
        skip = self.skip_prompt and self.next_tokens_are_prompt
        if not skip and value is not None:
            v = value[0] if value.ndim > 1 else value
            self.ids.extend(v.tolist())
        super().put(value)


def _apply_mode(key):
    remove_flash_codec(model)
    if key:
        apply_flash_codec(model, **MODES[key])


def _layer_kv_bytes(layer):
    if not getattr(layer, "is_initialized", False):
        return 0
    k, v = getattr(layer, "keys", None), getattr(layer, "values", None)
    if k is None or v is None or k.numel() == 0:
        return 0
    return (k.numel() + v.numel()) * k.element_size()


def _cache_bytes(cache):
    """cache 总占用 = codec 层归档存储 + 其余层 raw KV。返回 (total, codec 存储, codec 层 raw)。"""
    total = codec_st = codec_raw = 0
    for layer in cache.layers:
        if isinstance(layer, FlashCodecLayer):
            b = _layer_kv_bytes(layer)
            if b:
                codec_raw += b
                codec_st += layer.storage_bytes()
                total += layer.storage_bytes()
        else:
            total += _layer_kv_bytes(layer)
    return total, codec_st, codec_raw


def _stream_once(messages, mode_key, max_new, out):
    """生成一条回复（流式）。结果写入 out：ids/elapsed/cache。"""
    prompt = tok.apply_chat_template(messages, tokenize=False,
                                     add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    streamer = _IdStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    _apply_mode(mode_key)

    def _run():
        with torch.inference_mode():
            model.generate(input_ids=ids, max_new_tokens=max_new,
                           do_sample=False, pad_token_id=tok.eos_token_id,
                           streamer=streamer, use_cache=True)

    thread = threading.Thread(target=_run)
    t0 = time.time()
    thread.start()
    for piece in streamer:
        yield piece
    thread.join()
    out["elapsed"] = time.time() - t0
    out["ids"] = streamer.ids
    out["prompt_len"] = ids.shape[1]
    out["cache"] = None
    out["codec_layers"] = 0
    if mode_key is not None:
        info = getattr(model, "_flashcodec", None)
        cache = info["caches"][-1] if info and info["caches"] else None
        out["cache"] = cache
        out["codec_layers"] = len(cache.codec_layer_indices()) if cache else 0


def _fmt_mb(b):
    return f"{b / 1048576:.2f} MB"


def _make_plot(base, codec, mode_label):
    """两联柱状图：绝对 KV 存储 + 相对原版变化（%）。

    耗时变化按**每 token 耗时**（elapsed / n_new）比较——两侧可能因 EOS
    早停生成不同长度，直接比总耗时不公平。"""
    b_mb = base["total"] / 1048576
    c_mb = codec["total"] / 1048576
    d_store = (c_mb / b_mb - 1) * 100 if b_mb else 0.0
    b_t = base["elapsed"] / max(base["n_new"], 1)
    c_t = codec["elapsed"] / max(codec["n_new"], 1)
    d_time = (c_t / b_t - 1) * 100 if b_t > 0 else 0.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    fig.subplots_adjust(wspace=0.35)

    xs = ["原版 baseline", f"flashcodec\n{mode_label}"]
    bars = ax1.bar(xs, [b_mb, c_mb], color=["#8b93a3", "#4f8cff"], width=0.55)
    for bar, val in zip(bars, [b_mb, c_mb]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{val:.2f} MB", ha="center", va="bottom", fontsize=10)
    ax1.set_ylabel("full-attention 层 KV (MB)")
    ax1.set_title("KV Cache 存储")
    ax1.grid(axis="y", alpha=0.25)

    names = ["KV 存储", "每 token 耗时"]
    deltas = [d_store, d_time]
    colors = ["#3ecf8e" if d < 0 else "#ff6b6b" for d in deltas]
    bars2 = ax2.bar(names, deltas, color=colors, width=0.5)
    for bar, d in zip(bars2, deltas):
        y = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, y,
                 f"{d:+.1f}%", ha="center",
                 va="bottom" if y >= 0 else "top", fontsize=10)
    ax2.axhline(0, color="#8b93a3", lw=0.8)
    ax2.set_ylabel("相对原版变化 (%)")
    ax2.set_ylim(min(min(deltas) * 1.5, -10), max(max(deltas) * 1.5, 10))
    ax2.set_title("节约幅度（负数 = 更省）")
    ax2.grid(axis="y", alpha=0.25)

    s_store = (f"存储节约 {abs(d_store):.1f}%" if d_store < 0
               else f"存储增加 {d_store:.1f}%")
    s_time = (f"每token耗时节省 {abs(d_time):.1f}%" if d_time < 0
              else f"每token耗时增加 {d_time:.1f}%")
    fig.suptitle(f"{mode_label} · {s_store} · {s_time}", fontsize=11)
    fig.subplots_adjust(top=0.82)
    fig.canvas.draw()
    return fig, d_store, d_time


def _stats_md(side, s, d_store=None, d_time=None, match=None, exact=None):
    if side == "base":
        head = "**原版 baseline**"
        body = (f"KV Cache `{_fmt_mb(s['total'])}`（full-attn 层 raw，"
                f"{N_FULL} 层） · "
                f"{s['n_new']} tok / {s['elapsed']:.2f}s · "
                f"{s['tok_s']:.1f} tok/s · KV 长 {s['kv_len']}")
    else:
        head = f"**flashcodec · {s['mode_label']}**"
        ratio = (s['codec_st'] / s['codec_raw'] * 100) if s['codec_raw'] else 0
        body = (f"KV Cache `{_fmt_mb(s['total'])}`（codec 层归档 "
                f"{_fmt_mb(s['codec_st'])} = raw 的 {ratio:.1f}%，"
                f"{s['codec_layers']} 层） · "
                f"{s['n_new']} tok / {s['elapsed']:.2f}s · "
                f"{s['tok_s']:.1f} tok/s · KV 长 {s['kv_len']}")
        if d_store is not None:
            s_store = (f"存储 **节约 {abs(d_store):.1f}%**" if d_store < 0
                       else f"存储 **增加 {d_store:.1f}%**")
            s_time = (f"每 token 耗时 **节省 {abs(d_time):.1f}%**" if d_time < 0
                      else f"每 token 耗时 **增加 {d_time:.1f}%**")
            body += (f"\n\n**对比原版（按每 token 归一）：** {s_store}，{s_time}，"
                     f"token 一致率 **{match:.0%}**"
                     + ("（逐位一致 ✅）" if exact else "（有损分叉）"))
    return f"### {head}\n{body}"


def respond(user_msg, left_hist, right_hist, mode_label, max_new):
    user_msg = (user_msg or "").strip()
    if not user_msg:
        yield (gr.update(), gr.update(), gr.update(), gr.update(),
               gr.update(), gr.update())
        return

    left_hist = list(left_hist or []) + [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": ""},
    ]
    right_hist = list(right_hist or []) + [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": ""},
    ]
    yield (left_hist, right_hist, gr.update(), gr.update(), gr.update(),
           gr.update())

    messages = [{"role": m["role"], "content": m["content"]}
                for m in left_hist[:-1]]

    with gpu_lock:
        base_info, codec_info = {}, {}
        # 左：原版
        for piece in _stream_once(messages, None, max_new, base_info):
            left_hist[-1]["content"] += piece
            yield (gr.update(value=left_hist), gr.update(), gr.update(),
                   gr.update(), gr.update(), gr.update())
        # 右：flashcodec
        for piece in _stream_once(messages, mode_label, max_new, codec_info):
            right_hist[-1]["content"] += piece
            yield (gr.update(), gr.update(value=right_hist), gr.update(),
                   gr.update(), gr.update(), gr.update())
        remove_flash_codec(model)

        base_kv_len = base_info["prompt_len"] + max(len(base_info["ids"]) - 1, 0)
        b_total = _baseline_attn_bytes(base_kv_len)
        c_total, c_st, c_raw = _cache_bytes(codec_info["cache"])
        base = {"total": b_total, "elapsed": base_info["elapsed"],
                "n_new": len(base_info["ids"]),
                "tok_s": len(base_info["ids"]) / max(base_info["elapsed"], 1e-9),
                "kv_len": base_kv_len}
        codec = {"total": c_total, "codec_st": c_st, "codec_raw": c_raw,
                 "elapsed": codec_info["elapsed"],
                 "n_new": len(codec_info["ids"]),
                 "tok_s": len(codec_info["ids"]) / max(codec_info["elapsed"], 1e-9),
                 "kv_len": codec_info["cache"].get_seq_length() if codec_info["cache"] else None,
                 "codec_layers": codec_info.get("codec_layers", 0),
                 "mode_label": MODES[mode_label]["mode"]}

        fig, d_store, d_time = _make_plot(base, codec, mode_label)

        a, b = base_info["ids"], codec_info["ids"]
        n = min(len(a), len(b))
        match = (sum(x == y for x, y in zip(a[:n], b[:n])) / n) if n else 0.0
        exact = a == b and len(a) > 0

        md_l = _stats_md("base", base)
        md_r = _stats_md("codec", codec, d_store, d_time, match, exact)
        table = (
            "| 指标 | 原版 | flashcodec | 变化 |\n"
            "|---|---|---|---|\n"
            f"| KV Cache 存储 | {_fmt_mb(base['total'])} | "
            f"{_fmt_mb(codec['total'])} | {d_store:+.1f}% |\n"
            f"| 每 token 耗时 | {base['elapsed'] / max(base['n_new'], 1) * 1000:.1f} ms | "
            f"{codec['elapsed'] / max(codec['n_new'], 1) * 1000:.1f} ms | "
            f"{d_time:+.1f}% |\n"
            f"| 解码速度 | {base['tok_s']:.1f} tok/s | {codec['tok_s']:.1f} tok/s | "
            f"{(codec['tok_s'] / base['tok_s'] - 1) * 100 if base['tok_s'] else 0:+.1f}% |\n"
            f"| 新生成 token（总耗时） | {base['n_new']}（{base['elapsed']:.2f}s） | "
            f"{codec['n_new']}（{codec['elapsed']:.2f}s） | "
            f"一致率 {match:.0%}{'，逐位一致' if exact else ''} |")

    yield (gr.update(value=left_hist), gr.update(value=right_hist),
           md_l, md_r, gr.update(value=fig), table)


def clear_all():
    return [], [], "", "", None, ""


with gr.Blocks(title="flashcodec × Qwen3.5-0.8B") as demo:
    gr.Markdown(
        "# flashcodec × Qwen3.5-0.8B · 左右对比\n"
        "同一 prompt：左侧原生 KV Cache，右侧 flashcodec 编解码 KV Cache"
        "（JPEG 频域有损 / PNG 时域差分无损）。底部柱状图对比存储与每 token "
        "耗时的节约（两侧可能因 EOS 早停生成不同长度，按每 token 归一才公平）。")
    with gr.Row():
        left_bot = gr.Chatbot(label="原版 baseline（无压缩）", height=420)
        right_bot = gr.Chatbot(label="flashcodec（压缩 KV Cache）", height=420)
    with gr.Row():
        msg = gr.Textbox(placeholder="输入消息，Enter 发送（Shift+Enter 换行）",
                         show_label=False, scale=7, lines=2)
        send_btn = gr.Button("发 送", variant="primary", scale=1)
        clear_btn = gr.Button("清空", scale=1)
    with gr.Row():
        mode_dd = gr.Dropdown(choices=list(MODES), value=DEFAULT_MODE,
                              label="flashcodec 档位")
        max_sl = gr.Slider(32, 512, value=MAX_NEW, step=16,
                           label="最大生成 token 数")
    with gr.Row():
        stats_l = gr.Markdown()
        stats_r = gr.Markdown()
    with gr.Row():
        plot = gr.Plot(label="对比")
        table_md = gr.Markdown()

    inputs = [msg, left_bot, right_bot, mode_dd, max_sl]
    outputs = [left_bot, right_bot, stats_l, stats_r, plot, table_md]
    msg.submit(respond, inputs, outputs)
    send_btn.click(respond, inputs, outputs)
    clear_btn.click(clear_all, [], outputs)


if __name__ == "__main__":
    demo.launch(server_name=HOST, server_port=PORT)
