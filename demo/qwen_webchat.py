"""Qwen3.5-0.8B × flashcodec interactive web chat (zero third-party web deps, pure stdlib).

Usage: python demo/qwen_webchat.py   →   http://127.0.0.1:7860
Environment variables: FLASHCODEC_MODEL (default Qwen/Qwen3.5-0.8B), WEBCHAT_HOST,
WEBCHAT_PORT, WEBCHAT_NEW_TOKENS (default 256).

Features:
- Mode switching: baseline (native) / png keep=1.0 (lossless) / jpeg k=1,2,4 (lossy)
- Token-by-token streaming; every request goes through apply_flash_codec, the real user path
- Stats panel: codec layer count, cache storage vs raw, compression ratio, KV sequence
  length, tok/s
- Consistency check: run baseline and png lossless separately on the same prompt and
  assert the generated tokens are bitwise-identical
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

gpu_lock = threading.Lock()          # serialize requests on a single GPU


def apply_mode(key):
    remove_flash_codec(model)
    if MODES[key]:
        apply_flash_codec(model, **MODES[key])


def cache_stats(elapsed, n_new):
    """Collect cache stats for this generation (the patch-injected cache lives in the diagnostics slot)."""
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
    """Stream generation: apply_flash_codec injection + TextIteratorStreamer token by token."""
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
    """Consistency check: run baseline and the selected mode greedily once each and compare tokens."""
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

    def log_message(self, *args):                     # silence access logs
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
            self._json({"error": "messages must not be empty"}, 400)
            return

        if self.path == "/api/chat":
            mode = payload.get("mode", "baseline")
            if mode not in MODES:
                self._json({"error": f"unknown mode {mode}"}, 400)
                return
            with gpu_lock:
                self._sse_start()
                try:
                    for event in chat_stream(messages, mode):
                        self._sse(event)
                except Exception as e:                # a single request must not crash the server
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
<html lang="en">
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
    <span class="tag">KV cache: JPEG frequency-domain / PNG delta</span>
    <span class="tag" id="mdl"></span>
  </header>

  <div class="ctl">
    <label>Codec mode</label>
    <select id="mode">
      <option value="baseline">baseline (native, no compression)</option>
      <option value="png_lossless" selected>png keep=1.0 (lossless, bitwise-identical)</option>
      <option value="jpeg_k1">jpeg k=1 (8× lossy)</option>
      <option value="jpeg_k2">jpeg k=2 (4× lossy)</option>
      <option value="jpeg_k4">jpeg k=4 (2× lossy)</option>
    </select>
    <button id="verify">Verify lossless match</button>
    <button id="clear">Clear chat</button>
  </div>

  <div id="chat"></div>

  <textarea id="input" placeholder="Type a message, Enter to send (Shift+Enter for a newline)"></textarea>
  <div class="row">
    <button id="send">Send</button>
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
  div.innerHTML = `<div><div class="who">${role === 'user' ? 'You' : 'Qwen3.5 + flashcodec'}</div>` +
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
  const parts = [`codec layers <b>${s.codec_layers}</b>`,
                 `storage <b>${fmtBytes(s.storage)}</b>`];
  if (s.ratio != null)
    parts.push(`vs raw <b>${fmtBytes(s.raw)}</b> (<b>${(s.ratio*100).toFixed(1)}%</b>)`);
  if (s.kv_len != null) parts.push(`KV length <b>${s.kv_len}</b>`);
  parts.push(`speed <b>${s.tok_s}</b> tok/s`);
  return `<div class="stats">${parts.map(p => `<span>${p}</span>`).join('')}</div>`;
}

async function send() {
  if (busy) return;
  const text = inputEl.value.trim();
  if (!text) return;
  busy = true; sendBtn.disabled = true;
  inputEl.value = ''; statusEl.textContent = 'Generating…';
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
        else if (ev.type === 'error') { statusEl.textContent = 'Error: ' + ev.text; }
      }
    }
    history.push({ role: 'assistant', content: acc });
    statusEl.textContent = '';
  } catch (e) {
    statusEl.textContent = 'Request failed: ' + e;
  }
  busy = false; sendBtn.disabled = false; inputEl.focus();
}

async function verify() {
  if (busy) return;
  const text = inputEl.value.trim() || 'Tell me about West Lake.';
  busy = true; verifyBtn.disabled = true;
  statusEl.textContent = 'Verifying (running baseline and png lossless once each on the same prompt)…';
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
      ? '✅ Match: the PNG lossless mode generated tokens bitwise-identical to the baseline'
      : '❌ Mismatch: token divergence detected (please report this case)';
    div.innerHTML = `<div><div class="who">Consistency check · ${text}</div>` +
      `<div class="bubble"><span class="${cls}">${verdict}</span>\n` +
      `baseline : ${r.base_text}\ncodec    : ${r.codec_text}</div>` +
      statsHtml(r.stats) + `</div>`;
    chatEl.appendChild(div); scroll();
    statusEl.textContent = '';
  } catch (e) {
    statusEl.textContent = 'Verification failed: ' + e;
  }
  busy = false; verifyBtn.disabled = false;
}

sendBtn.onclick = send;
verifyBtn.onclick = verify;
clearBtn.onclick = () => { history = []; chatEl.innerHTML = ''; statusEl.textContent = 'Cleared (the server-side cache is rebuilt on the next generation)'; };
inputEl.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});
document.getElementById('mdl').textContent =
  'Greedy decoding · independent prefill per turn';
addMsg('ai', 'Hi! I am Qwen3.5-0.8B, and my KV cache is managed by flashcodec.\n' +
             'Try switching the mode above, or click "Verify lossless match" to see the PNG mode agree with the native one bit for bit.');
</script>
</body>
</html>
"""


if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    server.daemon_threads = True
    print(f"🌐 http://{HOST}:{PORT}  (Ctrl+C to quit)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nBye")
