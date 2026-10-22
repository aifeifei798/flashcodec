"""Qwen3.5-0.8B × flashcodec side-by-side Gradio demo.

Usage: python demo/qwen_gradio_chat.py   →   http://127.0.0.1:7860
Environment variables: FLASHCODEC_MODEL (default Qwen/Qwen3.5-0.8B), WEBCHAT_HOST,
WEBCHAT_PORT, WEBCHAT_NEW_TOKENS (default 128).

Two tabs:
1. **Chat** — the same prompt streamed on both sides (native KV cache vs the
   codec cache), with storage / latency / memory / quality stats underneath.
2. **Full comparison** — runs the baseline plus every codec tier once and
   reports, per tier: archive + view bytes, peak and steady GPU memory, decode
   tok/s, teacher-forced NLL of the continuation (a perplexity-style quality
   number), and the token match rate. Also draws storage / speed / NLL bars.

`Materialization-free decode` switches on `materialization_free=True`
(coefficient-domain attention + Triton kernel): the KV view is dropped after
prefill, so the steady memory column shows the archive only.
"""

import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
# The local model is loaded offline without a proxy; a socks proxy URL makes
# httpx/gradio fail at startup
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

from flashcodec import (FlashCodecCache, apply_flash_codec,
                        codec_attention_stats, remove_flash_codec)
from flashcodec.cache import FlashCodecLayer

MODEL = os.environ.get("FLASHCODEC_MODEL", "Qwen/Qwen3.5-0.8B")
HOST = os.environ.get("WEBCHAT_HOST", "127.0.0.1")
PORT = int(os.environ.get("WEBCHAT_PORT", "7860"))
MAX_NEW = int(os.environ.get("WEBCHAT_NEW_TOKENS", "128"))

# name -> (codec config, short label used in tables/plots)
MODES = {
    "png keep=1.0 · lossless (bitwise)": (
        dict(mode="png", block_size=8, keep_ratio=1.0), "png lossless"),
    "jpeg k=8 int8 · ~2× (safe)": (
        dict(mode="jpeg", block_size=8, keep_freqs=8, quant_bits=8), "k8 int8"),
    "jpeg k=4 · ~2×": (
        dict(mode="jpeg", block_size=8, keep_freqs=4), "k4"),
    "jpeg k=2 · ~4×": (
        dict(mode="jpeg", block_size=8, keep_freqs=2), "k2"),
    "jpeg k=2 int8 · ~7×": (
        dict(mode="jpeg", block_size=8, keep_freqs=2, quant_bits=8), "k2 int8"),
}
DEFAULT_MODE = "jpeg k=2 int8 · ~7×"

matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
matplotlib.rcParams["axes.unicode_minus"] = False

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading {MODEL} ({device})...")
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
print("Ready ✅")

gpu_lock = threading.Lock()

# The baseline's explicit DynamicCache is unusable on this hybrid architecture
# (linear_attention), so we derive the raw KV bytes of the full-attention layers
# from the config instead — flashcodec compresses only those layers, keeping the
# two sides apples-to-apples.
_dec = model.config.get_text_config(decoder=True)
_layer_types, _ = get_layer_types_and_kwargs(_dec)
N_FULL = sum(1 for t in _layer_types if t == "full_attention")
N_LAYERS = len(_layer_types)
_HKV = _dec.num_key_value_heads
_HEAD_DIM = getattr(_dec, "head_dim", _dec.hidden_size // _dec.num_attention_heads)
_ELEM = torch.empty((), dtype=model.dtype).element_size()


def _baseline_attn_bytes(kv_len):
    """Total raw KV bytes (K+V) of the original full-attention layers."""
    return N_FULL * 2 * kv_len * _HKV * _HEAD_DIM * _ELEM


class _IdStreamer(TextIteratorStreamer):
    """Record generated token ids (skipping the prompt frame) for match stats."""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.ids = []
        self.first_tok_time = None

    def put(self, value):
        skip = self.skip_prompt and self.next_tokens_are_prompt
        if not skip and value is not None:
            if self.first_tok_time is None:
                self.first_tok_time = time.time()
            v = value[0] if value.ndim > 1 else value
            self.ids.extend(v.tolist())
        super().put(value)


def _apply(mode_key, materialization_free=False):
    remove_flash_codec(model)
    if mode_key is not None:
        cfg = MODES[mode_key][0]
        apply_flash_codec(model, materialization_free=materialization_free,
                          **cfg)


def _apply_cfg(cfg, materialization_free=False):
    remove_flash_codec(model)
    if cfg is not None:
        apply_flash_codec(model, materialization_free=materialization_free,
                          **cfg)


def _layer_kv_bytes(layer):
    if not getattr(layer, "is_initialized", False):
        return 0
    k, v = getattr(layer, "keys", None), getattr(layer, "values", None)
    if k is None or v is None or k.numel() == 0:
        return 0
    return (k.numel() + v.numel()) * k.element_size()


def _cache_bytes(cache):
    """Total cache footprint = codec archive + materialized view + native layers.

    Returns ``(total, codec_storage, materialized_view, native_layers)``."""
    total = codec_st = view = native = 0
    for layer in cache.layers:
        if isinstance(layer, FlashCodecLayer):
            b = _layer_kv_bytes(layer)
            codec_st += layer.storage_bytes()
            if b:
                view += b
            total += layer.storage_bytes() + b
        else:
            b = _layer_kv_bytes(layer)
            native += b
            total += b
    return total, codec_st, view, native


def _continuation_nll(ids, prompt_len, cfg):
    """Teacher-forced NLL per token over the generated continuation (prefill
    semantics; the decode-only attention patch does not interfere)."""
    if ids.shape[1] <= prompt_len:
        return None
    remove_flash_codec(model)
    cache = FlashCodecCache.from_config(model.config, **cfg) if cfg else None
    with torch.inference_mode():
        out = model(input_ids=ids, past_key_values=cache,
                    use_cache=cache is not None)
    logits = out.logits[:, prompt_len - 1:-1].float()      # predicts prompt_len..
    tgt = ids[:, prompt_len:]
    logp = torch.log_softmax(logits, dim=-1)
    nll = -logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).mean().item()
    del out, logits, logp
    remove_flash_codec(model)
    return nll


def _timed_generate(prompt_ids, cfg, max_new, materialization_free):
    """Non-streaming generate with memory stats and a warmup so the Triton JIT
    is not charged to the measured run."""
    _apply_cfg(cfg, materialization_free)
    if device == "cuda":
        with torch.inference_mode():
            model.generate(input_ids=prompt_ids[:, :max(16, prompt_ids.shape[1] // 4)],
                           max_new_tokens=2, do_sample=False,
                           pad_token_id=tok.eos_token_id)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base_alloc = torch.cuda.memory_allocated()
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(input_ids=prompt_ids, max_new_tokens=max_new,
                             do_sample=False, pad_token_id=tok.eos_token_id)
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    peak = steady = 0.0
    if device == "cuda":
        peak = (torch.cuda.max_memory_allocated() - base_alloc) / 1048576
        steady = (torch.cuda.memory_allocated() - base_alloc) / 1048576
    info = getattr(model, "_flashcodec", None)
    cache = info["caches"][-1] if info and info["caches"] else None
    wrapped = info.get("codec_attention_modules", 0) if info else 0
    ids = out[0].tolist()
    remove_flash_codec(model)
    return dict(ids=ids, elapsed=dt, peak=peak, steady=steady,
                tok_s=len(ids) / max(dt, 1e-9), cache=cache,
                wrapped=wrapped)


def _tier_row(name, short, prompt_len, gen, base_ids):
    ids = gen["ids"]                                   # prompt + continuation
    kv_len = max(len(ids) - 1, 0)                      # last token has no KV
    if gen["cache"] is not None:
        total, storage, view, native = _cache_bytes(gen["cache"])
    else:
        total = _baseline_attn_bytes(kv_len)
        storage = view = native = 0
    raw = _baseline_attn_bytes(kv_len)
    n = min(len(ids), len(base_ids))
    match = (sum(a == b for a, b in zip(ids[:n], base_ids[:n])) / n) if n else 0.0
    exact = ids == base_ids and len(ids) > 0
    return dict(name=name, short=short, ids=ids, kv_len=kv_len, total=total,
                storage=storage, view=view, native=native, raw=raw,
                ratio=(total / raw) if raw else 1.0,
                tok_s=gen["tok_s"], elapsed=gen["elapsed"], peak=gen["peak"],
                steady=gen["steady"], match=match, exact=exact,
                wrapped=gen["wrapped"]), ids


def _fmt_mb(b):
    return f"{b / 1048576:.2f} MB"


def _make_plot(rows, base_name="raw KV"):
    """Storage / decode-speed bars, plus continuation NLL when available."""
    names = [r["short"] for r in rows]
    has_nll = any(r.get("nll") is not None for r in rows)
    n_panels = 3 if has_nll else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5.2 * n_panels, 3.9))
    fig.subplots_adjust(wspace=0.42, top=0.80)
    axes = list(axes) if n_panels > 1 else [axes]

    colors = ["#8b93a3"] + ["#4f8cff"] * (len(rows) - 1)

    vals = [r["total"] / 1048576 for r in rows]
    bars = axes[0].bar(names, vals, color=colors, width=0.6)
    for bar, v in zip(bars, vals):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    axes[0].set_ylabel("KV footprint (MB)")
    axes[0].set_title("Storage (archive + view + native)")
    axes[0].grid(axis="y", alpha=0.25)

    speeds = [r["tok_s"] for r in rows]
    bars = axes[1].bar(names, speeds, color=colors, width=0.6)
    for bar, v in zip(bars, speeds):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    axes[1].set_ylabel("decode tok/s")
    axes[1].set_title("Speed (total generate time)")
    axes[1].grid(axis="y", alpha=0.25)

    if has_nll:
        nlls = [r.get("nll") for r in rows]
        shown = [v if v is not None else 0.0 for v in nlls]
        bars = axes[2].bar(names, shown, color=colors, width=0.6)
        for bar, v, ok in zip(bars, shown, nlls):
            if ok is not None:
                axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                             f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        axes[2].set_ylabel("NLL / token")
        axes[2].set_title("Quality (continuation NLL, lower = better)")
        axes[2].grid(axis="y", alpha=0.25)
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=18, labelsize=8)

    fig.suptitle(f"{MODEL} · same prompt, per-tier comparison", fontsize=11)
    fig.canvas.draw()
    return fig


def _compare_table(rows):
    head = ("| Tier | KV total | vs raw | archive | view | "
            "tok/s | peak | steady | NLL | ΔNLL | match |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|\n")
    base = rows[0]
    out = [head]
    for r in rows:
        nll = f"{r['nll']:.3f}" if r.get("nll") is not None else "—"
        dnll = ("—" if r.get("nll") is None or base.get("nll") is None
                else f"{r['nll'] - base['nll']:+.3f}")
        match = "100% ✅" if r["exact"] else f"{r['match']:.0%}"
        out.append(
            f"| {r['name']} | {r['total'] / 1048576:.2f} MB | "
            f"{r['ratio'] * 100:.1f}% | {r['storage'] / 1048576:.2f} MB | "
            f"{r['view'] / 1048576:.2f} MB | "
            f"{r['tok_s']:.1f} | {r['peak']:.0f} MB | {r['steady']:.0f} MB | "
            f"{nll} | {dnll} | {match} |\n")
    out.append(
        "\nRows are full-attention layers only (the linear-attention layers keep "
        "their native state). `steady` is the memory still held after generate: "
        "the baseline's cache is freed on return, the codec cache is retained by "
        "the demo for inspection; `view` stays 0 with materialization-free "
        "decode.\n")
    return "".join(out)


def _stats_md(side, s, d_store=None, d_time=None, match=None, exact=None):
    if side == "base":
        return (f"### **Original baseline**\n"
                f"KV cache `{_fmt_mb(s['total'])}` (full-attn raw, {N_FULL} layers) · "
                f"{s['n_new']} tok / {s['elapsed']:.2f} s · {s['tok_s']:.1f} tok/s · "
                f"KV length {s['kv_len']}")
    head = f"### **flashcodec · {s['mode_label']}**"
    ratio = (s["codec_st"] / s["raw"] * 100) if s.get("raw") else 0
    body = (f"KV cache `{_fmt_mb(s['total'])}` (archive {_fmt_mb(s['codec_st'])} "
            f"= {ratio:.1f}% of raw · view {_fmt_mb(s['view'])} · "
            f"{s['codec_layers']}/{N_LAYERS} layers) · "
            f"{s['n_new']} tok / {s['elapsed']:.2f} s · {s['tok_s']:.1f} tok/s · "
            f"peak {s['peak']:.0f} MB / steady {s['steady']:.0f} MB · "
            f"KV length {s['kv_len']}")
    if d_store is not None:
        s_store = (f"storage **saved {abs(d_store):.1f}%**" if d_store < 0
                   else f"storage **increased {d_store:.1f}%**")
        s_time = (f"time per token **saved {abs(d_time):.1f}%**" if d_time < 0
                  else f"time per token **increased {d_time:.1f}%**")
        body += (f"\n\n**vs. baseline (per-token normalized):** {s_store}, "
                 f"{s_time}, token match **{match:.0%}**"
                 + (" (bitwise-identical ✅)" if exact else " (lossy divergence)"))
    return head + "\n" + body


def _stream_once(messages, mode_key, max_new, mf, out):
    """Generate one streamed reply; stats are written into ``out``."""
    prompt = tok.apply_chat_template(messages, tokenize=False,
                                     add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    streamer = _IdStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    _apply(mode_key, mf)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        out["base_alloc"] = torch.cuda.memory_allocated()

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
    out["ttft"] = ((streamer.first_tok_time - t0)
                   if streamer.first_tok_time else None)
    out["peak"] = out["steady"] = 0.0
    if device == "cuda":
        out["peak"] = (torch.cuda.max_memory_allocated()
                       - out["base_alloc"]) / 1048576
        out["steady"] = (torch.cuda.memory_allocated()
                         - out["base_alloc"]) / 1048576
    info = getattr(model, "_flashcodec", None)
    out["cache"] = info["caches"][-1] if info and info["caches"] else None
    out["codec_layers"] = (len(out["cache"].codec_layer_indices())
                           if out["cache"] else 0)
    out["wrapped"] = codec_attention_stats(model)


def respond(user_msg, left_hist, right_hist, mode_label, max_new, mf):
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
        for piece in _stream_once(messages, None, max_new, False, base_info):
            left_hist[-1]["content"] += piece
            yield (gr.update(value=left_hist), gr.update(), gr.update(),
                   gr.update(), gr.update(), gr.update())
        for piece in _stream_once(messages, mode_label, max_new, mf, codec_info):
            right_hist[-1]["content"] += piece
            yield (gr.update(), gr.update(value=right_hist), gr.update(),
                   gr.update(), gr.update(), gr.update())

        base_kv_len = base_info["prompt_len"] + max(len(base_info["ids"]) - 1, 0)
        b_total = _baseline_attn_bytes(base_kv_len)
        c_total, c_st, c_view, c_native = _cache_bytes(codec_info["cache"])
        base = {"total": b_total, "elapsed": base_info["elapsed"],
                "n_new": len(base_info["ids"]),
                "tok_s": len(base_info["ids"]) / max(base_info["elapsed"], 1e-9),
                "kv_len": base_kv_len}
        codec = {"total": c_total, "codec_st": c_st, "view": c_view,
                 "native": c_native, "raw": b_total,
                 "elapsed": codec_info["elapsed"],
                 "n_new": len(codec_info["ids"]),
                 "tok_s": len(codec_info["ids"]) / max(codec_info["elapsed"], 1e-9),
                 "kv_len": (codec_info["cache"].get_seq_length()
                            if codec_info["cache"] else None),
                 "codec_layers": codec_info.get("codec_layers", 0),
                 "peak": codec_info["peak"], "steady": codec_info["steady"],
                 "mode_label": mode_label}

        b_t = base["elapsed"] / max(base["n_new"], 1)
        c_t = codec["elapsed"] / max(codec["n_new"], 1)
        d_store = (c_total / b_total - 1) * 100 if b_total else 0.0
        d_time = (c_t / b_t - 1) * 100 if b_t > 0 else 0.0

        fig = _make_plot([
            dict(short="raw KV", total=b_total, tok_s=base["tok_s"]),
            dict(short=mode_label.split(" · ")[0], total=c_total,
                 tok_s=codec["tok_s"]),
        ])

        a, b = base_info["ids"], codec_info["ids"]
        n = min(len(a), len(b))
        match = (sum(x == y for x, y in zip(a[:n], b[:n])) / n) if n else 0.0
        exact = a == b and len(a) > 0

        md_l = _stats_md("base", base)
        md_r = _stats_md("codec", codec, d_store, d_time, match, exact)
        table = (
            "| Metric | Baseline | flashcodec | Δ |\n"
            "|---|---|---|---|\n"
            f"| KV footprint | {_fmt_mb(b_total)} | {_fmt_mb(c_total)} | "
            f"{d_store:+.1f}% |\n"
            f"| archive / native | — / {_fmt_mb(b_total)} | "
            f"{_fmt_mb(c_st)} / {_fmt_mb(c_native)} | — |\n"
            f"| materialized view | — | {_fmt_mb(c_view)} | — |\n"
            f"| peak memory | {base_info['peak']:.0f} MB | "
            f"{codec_info['peak']:.0f} MB | — |\n"
            f"| steady memory | {base_info['steady']:.0f} MB | "
            f"{codec_info['steady']:.0f} MB | — |\n"
            f"| time per token | {b_t * 1000:.1f} ms | {c_t * 1000:.1f} ms | "
            f"{d_time:+.1f}% |\n"
            f"| decode speed | {base['tok_s']:.1f} tok/s | "
            f"{codec['tok_s']:.1f} tok/s | "
            f"{(codec['tok_s'] / base['tok_s'] - 1) * 100 if base['tok_s'] else 0:+.1f}% |\n"
            f"| new tokens (total) | {base['n_new']} ({base['elapsed']:.2f} s) | "
            f"{codec['n_new']} ({codec['elapsed']:.2f} s) | "
            f"match {match:.0%}{', bitwise' if exact else ''} |")
        if mf:
            table += (f"\n| materialization-free | — | "
                      f"{codec_info['wrapped']}/{N_LAYERS} attention modules | — |")

    yield (gr.update(value=left_hist), gr.update(value=right_hist),
           md_l, md_r, gr.update(value=fig), table)


def compare_all(prompt_text, max_new, mf):
    prompt_text = (prompt_text or "").strip()
    if not prompt_text:
        yield gr.update(), gr.update(), "Enter a prompt first."
        return
    messages = [{"role": "user", "content": prompt_text}]
    prompt = tok.apply_chat_template(messages, tokenize=False,
                                     add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = ids.shape[1]

    with gpu_lock:
        base = _timed_generate(ids, None, max_new, False)
        base_row, _ = _tier_row("baseline (raw KV)", "raw KV", prompt_len,
                                base, base["ids"])
        base_row["nll"] = _continuation_nll(
            torch.tensor([base["ids"]], device=device), prompt_len, None)
        rows = [base_row]

        for name, (cfg, short) in MODES.items():
            gen = _timed_generate(ids, cfg, max_new, mf)
            row, _ = _tier_row(name, short, prompt_len, gen, base["ids"])
            row["nll"] = _continuation_nll(
                torch.tensor([row["ids"]], device=device), prompt_len, cfg)
            rows.append(row)
            yield (gr.update(), gr.update(value=_compare_table(rows)),
                   f"done: {short} …")

    fig = _make_plot(rows)
    note = (f"`{prompt_len}` prompt tokens, up to `{max_new}` new tokens; "
            "NLL is teacher-forced on the continuation (lower = better), "
            "match is against the baseline reply. "
            + ("Materialization-free decode ON." if mf else
               "Materialization-free decode OFF (view kept)."))
    yield gr.update(value=fig), gr.update(value=_compare_table(rows)), note


def clear_all():
    return [], [], "", "", None, ""


SAMPLE_PROMPT = ("Explain how a KV cache works in transformer inference, and "
                 "why its memory grows with context length.")


with gr.Blocks(title="flashcodec × Qwen3.5-0.8B") as demo:
    gr.Markdown(
        "# flashcodec × Qwen3.5-0.8B\n"
        f"Compresses the KV cache of the {N_FULL}/{N_LAYERS} full-attention "
        "layers (linear-attention layers keep their native state). "
        "**Chat** streams the same prompt on both sides; **Full comparison** "
        "runs every tier and reports storage, peak/steady memory, decode speed, "
        "continuation NLL and token match.")

    with gr.Tabs():
        with gr.Tab("Chat"):
            with gr.Row():
                left_bot = gr.Chatbot(
                    label="Original baseline (no compression)", height=380)
                right_bot = gr.Chatbot(
                    label="flashcodec (compressed KV cache)", height=380)
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Type a message, Enter to send "
                                "(Shift+Enter for a newline)",
                    show_label=False, scale=7, lines=2)
                send_btn = gr.Button("Send", variant="primary", scale=1)
                clear_btn = gr.Button("Clear", scale=1)
            with gr.Row():
                mode_dd = gr.Dropdown(choices=list(MODES), value=DEFAULT_MODE,
                                      label="flashcodec mode", scale=6)
                mf_ck = gr.Checkbox(
                    value=False, label="Materialization-free decode (Triton)",
                    scale=3)
                max_sl = gr.Slider(32, 512, value=MAX_NEW, step=16,
                                   label="Max new tokens", scale=3)
            with gr.Row():
                stats_l = gr.Markdown()
                stats_r = gr.Markdown()
            with gr.Row():
                plot = gr.Plot(label="Comparison")
            table_md = gr.Markdown()

            inputs = [msg, left_bot, right_bot, mode_dd, max_sl, mf_ck]
            outputs = [left_bot, right_bot, stats_l, stats_r, plot, table_md]
            msg.submit(respond, inputs, outputs)
            send_btn.click(respond, inputs, outputs)
            clear_btn.click(clear_all, [], outputs)

        with gr.Tab("Full comparison"):
            gr.Markdown(
                "Runs the baseline and every codec tier on the same prompt. "
                "The baseline is generated first; each tier is warmed up once "
                "so the Triton JIT is not charged to the measured run.")
            cmp_prompt = gr.Textbox(value=SAMPLE_PROMPT, lines=3,
                                    label="Prompt")
            with gr.Row():
                cmp_max = gr.Slider(32, 512, value=MAX_NEW, step=16,
                                    label="Max new tokens")
                cmp_mf = gr.Checkbox(value=True,
                                     label="Materialization-free decode (Triton)")
                cmp_btn = gr.Button("Run comparison", variant="primary")
            cmp_plot = gr.Plot(label="Per-tier comparison")
            cmp_note = gr.Markdown()
            cmp_table = gr.Markdown()
            cmp_btn.click(compare_all, [cmp_prompt, cmp_max, cmp_mf],
                          [cmp_plot, cmp_table, cmp_note])


if __name__ == "__main__":
    demo.launch(server_name=HOST, server_port=PORT)
