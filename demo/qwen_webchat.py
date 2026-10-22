"""Qwen3.5-0.8B × flashcodec 互动网页 Chat（零第三方 Web 依赖，纯 stdlib）。

用法：python demo/qwen_webchat.py   →   http://127.0.0.1:7860
环境变量：FLASHCODEC_MODEL（默认 Qwen/Qwen3.5-0.8B）、WEBCHAT_HOST、
WEBCHAT_PORT、WEBCHAT_NEW_TOKENS（默认 256）。

功能：
- 模式切换：baseline（原生）/ png keep=1.0（无损）/ jpeg k=1,2,4（有损）
- 逐 token 流式输出；每次请求走 apply_flash_codec 真实用户路径
- 统计面板：codec 层数、缓存存储 vs raw、压缩比、KV 序列长、tok/s
- 一致性校验：同 prompt 分别跑 baseline 与 png 无损，断言生成 token 逐位一致
"""

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          TextIteratorStreamer)

from flashcodec import apply_flash_codec, remove_flash_codec

MODEL = os.environ.get("FLASHCODEC_MODEL", "Qwen/Qwen3.5-0.8B")
HOST = os.environ.get("WEBCHAT_HOST", "127.0.0.1")
PORT = int(os.environ.get("WEBCHAT_PORT", "7860"))
MAX_NEW = int(os.environ.get("WEBCHAT_NEW_TOKENS", "256"))

MODES = {
    "baseline":   None,
    "png_lossless": dict(mode="png", block_size=8, keep_ratio=1.0),
    "jpeg_k1":    dict(mode="jpeg", block_size=8, keep_freqs=1),
    "jpeg_k2":    dict(mode="jpeg", block_size=8, keep_freqs=2),
    "jpeg_k4":    dict(mode="jpeg", block_size=8, keep_freqs=4),
}

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

gpu_lock = threading.Lock()          # 单卡串行化请求


def apply_mode(key):
    remove_flash_codec(model)
    if MODES[key]:
        apply_flash_codec(model, **MODES[key])


def cache_stats(elapsed, n_new):
    """收集本次生成的缓存统计（patch 注入的 cache 保存在诊断槽里）。"""
    info = getattr(model, "_flashcodec", None)
    cache = info["caches"][-1] if info and info["caches"] else None
    if cache is None or not cache.layers:
        return {"mode": "baseline", "codec_layers": 0, "storage": 0,
                "raw": 0, "ratio": None, "kv_len": None,
                "tok_s": round(n_new / elapsed, 1) if elapsed > 0 else None}
    idxs = cache.codec_layer_indices()
    raw = sum(layer.keys.numel() * layer.keys.element_size() * 2
              for layer in (cache.layers[i] for i in idxs)
              if layer.is_initialized and layer.keys.numel() > 0)
    st = cache.storage_bytes()
    return {"mode": info["mode"], "codec_layers": len(idxs),
            "storage": st, "raw": raw,
            "ratio": round(st / raw, 4) if raw else None,
            "kv_len": cache.get_seq_length(),
            "tok_s": round(n_new / elapsed, 1) if elapsed > 0 else None}


def chat_stream(messages, mode):
    """流式生成：apply_flash_codec 注入 + TextIteratorStreamer 逐 token。"""
    prompt = tok.apply_chat_template(messages, tokenize=False,
                                     add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    streamer = TextIteratorStreamer(tok, skip_prompt=True,
                                    skip_special_tokens=True)
    apply_mode(mode)

    def _run():
        with torch.no_grad():
            model.generate(input_ids=ids, max_new_tokens=MAX_NEW,
                           do_sample=False, pad_token_id=tok.eos_token_id,
                           streamer=streamer, use_cache=True)

    thread = threading.Thread(target=_run)
    t0 = time.time()
    thread.start()
    n_chars = 0
    for piece in streamer:
        n_chars += len(piece)
        yield {"type": "token", "text": piece}
    thread.join()
    elapsed = time.time() - t0
    yield {"type": "stats", **cache_stats(elapsed, MAX_NEW),
           "elapsed": round(elapsed, 2)}


def verify_pair(messages, mode="png_lossless", n_new=48):
    """一致性校验：baseline 与所选档位各跑一次贪心，比较生成 token。"""
    prompt = tok.apply_chat_template(messages, tokenize=False,
                                     add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    kw = dict(input_ids=ids, max_new_tokens=n_new, do_sample=False,
              pad_token_id=tok.eos_token_id, use_cache=True)
    with gpu_lock:
        with torch.no_grad():
            apply_mode("baseline")
            ref = model.generate(**kw)
            stats_base = cache_stats(0.0, n_new)
            apply_mode(mode)
            out = model.generate(**kw)
            stats_codec = cache_stats(0.0, n_new)
        remove_flash_codec(model)
    ref_new = ref[0, ids.shape[1]:]
    out_new = out[0, ids.shape[1]:]
    return {
        "exact": bool(torch.equal(ref_new, out_new)),
        "base_text": tok.decode(ref_new, skip_special_tokens=True),
        "codec_text": tok.decode(out_new, skip_special_tokens=True),
        "base_tokens": int(ref_new.numel()),
        "codec_tokens": int(out_new.numel()),
        "stats": stats_codec,
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):                     # 静默访问日志
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _sse_start(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

    def _sse(self, obj):
        self.wfile.write(f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"
                         .encode())
        self.wfile.flush()

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._json({"error": "bad json"}, 400)
            return
        messages = payload.get("messages") or []
        if not messages:
            self._json({"error": "messages 为空"}, 400)
            return

        if self.path == "/api/chat":
            mode = payload.get("mode", "baseline")
            if mode not in MODES:
                self._json({"error": f"未知模式 {mode}"}, 400)
                return
            with gpu_lock:
                self._sse_start()
                try:
                    for event in chat_stream(messages, mode):
                        self._sse(event)
                except Exception as e:                # 服务不因单次请求崩溃
                    self._sse({"type": "error", "text": repr(e)})
        elif self.path == "/api/verify":
            mode = payload.get("mode", "png_lossless")
            n_new = int(payload.get("n_new", 48))
            try:
                self._json(verify_pair(messages, mode, n_new))
            except Exception as e:
                self._json({"error": repr(e)}, 500)
        else:
            self._json({"error": "not found"}, 404)


PAGE = r"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>flashcodec × Qwen3.5-0.8B</title>
<style>
  :root { --bg:#0f1115; --panel:#171a21; --line:#2a2f3a; --txt:#e6e9ef;
          --dim:#8b93a3; --acc:#4f8cff; --ok:#3ecf8e; --bad:#ff6b6b; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--txt); font:15px/1.6
         -apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif; }
  .wrap { max-width:960px; margin:0 auto; padding:20px 16px 40px; }
  header { display:flex; align-items:center; gap:12px; flex-wrap:wrap;
           border-bottom:1px solid var(--line); padding-bottom:14px; }
  h1 { font-size:18px; margin:0; }
  .tag { font-size:12px; color:var(--dim); border:1px solid var(--line);
         border-radius:999px; padding:2px 10px; }
  .ctl { display:flex; gap:10px; align-items:center; margin:14px 0;
         flex-wrap:wrap; }
  select, button { background:var(--panel); color:var(--txt); border:1px solid
         var(--line); border-radius:8px; padding:7px 12px; font-size:14px;
         cursor:pointer; }
  button:hover { border-color:var(--acc); }
  button:disabled { opacity:.45; cursor:not-allowed; }
  #chat { background:var(--panel); border:1px solid var(--line);
          border-radius:12px; padding:16px; min-height:320px;
          max-height:60vh; overflow-y:auto; }
  .msg { margin:10px 0; display:flex; }
  .msg .bubble { max-width:82%; padding:9px 13px; border-radius:12px;
                 white-space:pre-wrap; word-break:break-word; }
  .msg.user { justify-content:flex-end; }
  .msg.user .bubble { background:#24406e; }
  .msg.ai .bubble { background:#1e2330; border:1px solid var(--line); }
  .msg .who { font-size:12px; color:var(--dim); margin-bottom:2px; }
  .stats { font-size:12px; color:var(--dim); margin:6px 0 0;
           display:flex; gap:16px; flex-wrap:wrap; }
  .stats b { color:var(--txt); font-weight:600; }
  .ok { color:var(--ok); } .bad { color:var(--bad); }
  #input { width:100%; margin-top:12px; background:var(--panel); color:var(--txt);
           border:1px solid var(--line); border-radius:10px; padding:11px 13px;
           font-size:15px; resize:vertical; min-height:64px; }
  .row { display:flex; gap:10px; margin-top:10px; align-items:center; }
  #status { font-size:13px; color:var(--dim); }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>flashcodec × Qwen3.5-0.8B</h1>
    <span class="tag">KV Cache: JPEG 频域 / PNG 差分</span>
    <span class="tag" id="mdl"></span>
  </header>

  <div class="ctl">
    <label>编码档位</label>
    <select id="mode">
      <option value="baseline">baseline（原生，无压缩）</option>
      <option value="png_lossless" selected>png keep=1.0（无损，逐位一致）</option>
      <option value="jpeg_k1">jpeg k=1（8× 有损）</option>
      <option value="jpeg_k2">jpeg k=2（4× 有损）</option>
      <option value="jpeg_k4">jpeg k=4（2× 有损）</option>
    </select>
    <button id="verify">无损一致性校验</button>
    <button id="clear">清空对话</button>
  </div>

  <div id="chat"></div>

  <textarea id="input" placeholder="输入消息，Enter 发送（Shift+Enter 换行）"></textarea>
  <div class="row">
    <button id="send">发 送</button>
    <span id="status"></span>
  </div>
</div>

<script>
const chatEl = document.getElementById('chat');
const inputEl = document.getElementById('input');
const sendBtn = document.getElementById('send');
const verifyBtn = document.getElementById('verify');
const clearBtn = document.getElementById('clear');
const modeEl = document.getElementById('mode');
const statusEl = document.getElementById('status');
let history = [];
let busy = false;

function scroll() { chatEl.scrollTop = chatEl.scrollHeight; }

function addMsg(role, text) {
  const div = document.createElement('div');
  div.className = 'msg ' + (role === 'user' ? 'user' : 'ai');
  div.innerHTML = `<div><div class="who">${role === 'user' ? '你' : 'Qwen3.5 + flashcodec'}</div>` +
                  `<div class="bubble"></div></div>`;
  div.querySelector('.bubble').textContent = text;
  chatEl.appendChild(div); scroll();
  return div.querySelector('.bubble');
}

function fmtBytes(b) {
  if (b === 0) return '0';
  return (b / 1048576).toFixed(2) + ' MB';
}

function statsHtml(s) {
  const parts = [`codec 层 <b>${s.codec_layers}</b>`,
                 `存储 <b>${fmtBytes(s.storage)}</b>`];
  if (s.ratio != null)
    parts.push(`vs raw <b>${fmtBytes(s.raw)}</b>（<b>${(s.ratio*100).toFixed(1)}%</b>）`);
  if (s.kv_len != null) parts.push(`KV 长 <b>${s.kv_len}</b>`);
  parts.push(`速度 <b>${s.tok_s}</b> tok/s`);
  return `<div class="stats">${parts.map(p => `<span>${p}</span>`).join('')}</div>`;
}

async function send() {
  if (busy) return;
  const text = inputEl.value.trim();
  if (!text) return;
  busy = true; sendBtn.disabled = true;
  inputEl.value = ''; statusEl.textContent = '生成中…';
  addMsg('user', text);
  history.push({ role: 'user', content: text });
  const bubble = addMsg('ai', '');
  let acc = '';
  try {
    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: history, mode: modeEl.value }),
    });
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf('\n\n')) >= 0) {
        const raw = buf.slice(0, idx); buf = buf.slice(idx + 2);
        if (!raw.startsWith('data: ')) continue;
        const ev = JSON.parse(raw.slice(6));
        if (ev.type === 'token') { acc += ev.text; bubble.textContent = acc; scroll(); }
        else if (ev.type === 'stats') { bubble.parentElement.insertAdjacentHTML('beforeend', statsHtml(ev)); scroll(); }
        else if (ev.type === 'error') { statusEl.textContent = '错误：' + ev.text; }
      }
    }
    history.push({ role: 'assistant', content: acc });
    statusEl.textContent = '';
  } catch (e) {
    statusEl.textContent = '请求失败：' + e;
  }
  busy = false; sendBtn.disabled = false; inputEl.focus();
}

async function verify() {
  if (busy) return;
  const text = inputEl.value.trim() || '介绍一下西湖。';
  busy = true; verifyBtn.disabled = true;
  statusEl.textContent = '校验中（同 prompt 跑 baseline 与 png 无损各一次）…';
  try {
    const resp = await fetch('/api/verify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: [{ role: 'user', content: text }],
                             mode: 'png_lossless', n_new: 48 }),
    });
    const r = await resp.json();
    const div = document.createElement('div');
    div.className = 'msg ai';
    const cls = r.exact ? 'ok' : 'bad';
    const verdict = r.exact
      ? '✅ 一致：PNG 无损档生成 token 与 baseline 逐位完全相同'
      : '❌ 不一致：出现 token 分歧（请反馈此场景）';
    div.innerHTML = `<div><div class="who">一致性校验 · ${text}</div>` +
      `<div class="bubble"><span class="${cls}">${verdict}</span>\n` +
      `baseline : ${r.base_text}\ncodec    : ${r.codec_text}</div>` +
      statsHtml(r.stats) + `</div>`;
    chatEl.appendChild(div); scroll();
    statusEl.textContent = '';
  } catch (e) {
    statusEl.textContent = '校验失败：' + e;
  }
  busy = false; verifyBtn.disabled = false;
}

sendBtn.onclick = send;
verifyBtn.onclick = verify;
clearBtn.onclick = () => { history = []; chatEl.innerHTML = ''; statusEl.textContent = '已清空（服务端缓存随下次生成重建）'; };
inputEl.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});
document.getElementById('mdl').textContent =
  '贪心解码 · 每轮独立 prefill';
addMsg('ai', '你好！我是 Qwen3.5-0.8B，KV Cache 由 flashcodec 编解码管理。\n' +
             '试试切换上方档位，或点「无损一致性校验」看 PNG 档与原生逐位一致。');
</script>
</body>
</html>
"""


if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    server.daemon_threads = True
    print(f"🌐 http://{HOST}:{PORT}  （Ctrl+C 退出）")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n再见")
