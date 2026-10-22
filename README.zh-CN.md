# flashcodec

> 把 Transformer 的时序 KV Cache 张量当作**高维多媒体图像信号**来压缩。

[English](README.md) | [简体中文](README.zh-CN.md)

传统 LLM 研究把隐空间张量视为孤立标量因子；flashcodec 认为 `(B, H, S, D)`
布局里的 **seq 轴是一幅高维图像的行/列**——相邻 token 隐向量强自相关，
能量集中在低频、差分后熵骤降。于是数字多媒体领域的两大经典压缩理论可以
直接迁移：

| 模式 | 灵感 | 变换 | 性质 | 存储 |
|---|---|---|---|---|
| `mode="jpeg"` | JPEG | 分块正交 1D DCT-II，保留最低 K 个低频系数 | **有损频域** | `K/r`（4×~8× 显存节省） |
| `mode="png"` | PNG | 块首锚点（I-Token）+ 一阶差分 Δ，Top-M 裁剪 | **无损/近无损时域** | 稠密等大；残差稀疏性留给熵编码 |

数学形式（块长 r，操作轴 dim=-2）：

- **JPEG**：$X_{\text{blk}} \in \mathbb{R}^{r \times D}$，$C = M_{\text{enc}} X_{\text{blk}}$，
  $M_{\text{enc}} \in \mathbb{R}^{K \times r}$，$M[j,n] = c_j \cos\frac{\pi}{r}(n+\tfrac12)j$，
  行正交归一（$K=r$ 时逐块无损）；重构 $\hat X = M_{\text{enc}}^\top C$。
- **PNG**：$\Delta_i = x_i - x_{i-1}$（$i \ge 1$），$x_0$ 为全精度锚点；
  重建 $\hat x_i = x_0 + \sum_{j=1}^{i} \Delta_j$（`cumsum`，纯线性、autograd 可微）。
  相邻隐向量自相关 ⇒ $\mathrm{Var}(\Delta)/\mathrm{Var}(X) \approx 0.03$（**30× 方差压缩**，
  实测），Top-M 裁剪后残差近零稀疏。

```
                    ┌────────────────────────────────────────────┐
  (B,H,S,D) ──append──▶  raw_buffer（未满 r 的尾部，原始高保真）  │
                    │        │ 满 r 个 token，批量归档（一次 GEMM， │
                    │        ▼  原地切片写入）                    │
                    │  compressed_storage（倍增预分配）            │
                    │   jpeg: 系数 (B,H,capacity,K,D)             │
                    │   png:  deltas (B,H,capacity,r-1,D)         │
                    │        + anchors (B,H,capacity,D)           │
                    └───────────────┬────────────────────────────┘
                                    │ read()：有效区单次解码 + raw 尾
                                    ▼
                        重构 KV 视图 (B,H,S,D)
```

归档存放在**单块几何倍增缓冲区**里（std::vector 式：初始 16 块，装满翻倍）。
每满块只做一次原地切片写入（无反复重新分配）；`read()` 对有效区做单次批量
GEMM / cumsum。若改成逐块碎片归档 + 逐归档循环解压，S=4096 的逐步解码
read 会因 kernel launch 开销慢约 170×（实测 31ms → 0.18ms）。

## 因果安全设计

只归档**完整块**；当前未满块永远留在 `raw_buffer`（原始精度）。增量解码时
任意时刻的未来扰动都不会改写已归档张量（测试逐位验证）。`mode="png",
keep_ratio=1.0` 的无损档位下，逐步解码注意力与全量前向注意力对齐 ≤1e-4。

## 安装

```bash
pip install -e .        # 依赖：torch>=2.0
```

## 快速上手

```python
import torch
from flashcodec import FlashCodecCache, apply_flash_codec

# 1) 原生：transformers 5.x 一等公民 Cache 后端（任意 CausalLM）
cache = FlashCodecCache(mode="jpeg", block_size=8, keep_freqs=2)  # 4×
out = model.generate(input_ids, max_new_tokens=64,
                     past_key_values=cache, use_cache=True)
print(cache.get_seq_length(), cache.storage_bytes())

# 2) 一行热补丁：之后每次 generate() 默认走编解码缓存
apply_flash_codec(model, mode="png", block_size=8, keep_ratio=1.0)
out = model.generate(input_ids, max_new_tokens=64)
# PNG 无损档位与未打补丁模型的生成 token 逐位完全一致
from flashcodec import remove_flash_codec
remove_flash_codec(model)          # 随时还原默认行为

# 3) 独立使用（无需 transformers）：直接操作缓存管理器
from flashcodec import CodecKVCache
cache = CodecKVCache(mode="jpeg", block_size=8, keep_freqs=2)
k, v = cache.forward(k_seq, v_seq)          # 全量前向填充
k, v = cache.decode_step(new_k, new_v)      # 增量解码
```

`FlashCodecCache` 继承 `transformers.cache_utils.Cache`（5.x layer API）：
KV 写入与读取的替换收敛在 `cache.update(k, v, layer_idx)` 接口——零模型
手术，任意 CausalLM（含 Llama/Qwen/Mistral 风格 GQA）通用。

## Transformers 集成（Qwen3.5-0.8B 实战验证）

混合架构（如 Qwen3.5：18 层线性注意力 + 6 层全注意力）自动处理——
`FlashCodecCache.from_config(model.config)` **只给 `full_attention` 层挂
codec 层**，其余层类型保留原生动态层（递归/卷积状态不受影响）。
`apply_flash_codec` 在拿到模型 config 时自动走 `from_config`。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from flashcodec import apply_flash_codec

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B",
                                             torch_dtype="bfloat16").cuda()
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
ids = tok("杭州的西湖", return_tensors="pt").input_ids.cuda()

# 一行搞定——之后每次 generate() 都走编解码缓存
apply_flash_codec(model, mode="jpeg", block_size=8, keep_freqs=2)
out = model.generate(ids, max_new_tokens=48, do_sample=False)
```

Qwen3.5-0.8B 实测（bf16、CUDA、48 个贪心 token，`benchmarks/run_qwen_demo.py`）：

| 档位 | 存储 vs raw KV | 输出 | 结论 |
|---|---|---|---|
| `png keep=1.0` | ~1.9×（fp32 差分换精确）| **生成 token 与未打补丁模型逐位一致** | 100% 保真基线 |
| `jpeg k=2` | **27.6%**（约 4×）| 文本流畅，token 一致率 28.8% | 有损档 |
| `jpeg k=4` | 51.7%（约 2×）| 文本流畅，token 一致率 35.6% | 有损档 |

混合层审计：`codec_layers=6/24`（仅全注意力层），零警告。复跑：
`python benchmarks/run_qwen_demo.py`（离线加载；支持 `FLASHCODEC_MODEL` /
`FLASHCODEC_NEW_TOKENS` 环境变量）。

dtype 契约（详见 `AGENTS.md`）：PNG 差分恒为 fp32（bf16 差分会破坏无损
档承诺的逐位精确性）；JPEG 系数与 `read()` 视图保持流输入 dtype
（SDPA 要求 q/k/v 同 dtype）。

## 基准（实测，B=2 H=8 S=2048 D=64，随机游走自相关信号，fp32）

### 重构质量（`benchmarks/bench_reconstruct.py`）

| 配置 | PSNR (dB) | cosine | 存储比 |
|---|---|---|---|
| jpeg r=8 k=1（8×） | 42.95 | 0.99943 | 12.5% |
| jpeg r=8 k=2（4×） | 47.22 | 0.99985 | 25.0% |
| jpeg r=8 k=4（2×） | 51.66 | 1.00000 | 50.0% |
| png r=8 keep=0.5 | 50.11 | 0.99997 | 100%（稀疏留熵编码）|
| png r=8 keep=1.0 | ~196（fp 舍入级）| 1.00000 | 100% |

### 缓存占用与耗时（`benchmarks/bench_vram.py`，CPU）

| seq | 模式 | storage | 原始比 | fill | read |
|---|---|---|---|---|---|
| 4096 | jpeg k=2 | 4.0MB | 0.25 | 15.1ms | 2.0ms |
| 16384 | jpeg k=2 | 16.0MB | 0.25 | 29.8ms | 13.4ms |
| 65536 | jpeg k=2 | 64.0MB | 0.25 | 103.9ms | 38.7ms |
| 65536 | png keep=1.0 | 256.0MB | 1.00 | 209.5ms | 71.9ms |

PNG 模式不省稠密存储（无损基线的宿命）——它的价值在 30× 方差压缩 + Top-M
近零稀疏残差，是熵编码器的理想输入（本库不内置熵编码）。

## 目录结构

```
flashcodec/
├── flashcodec/
│   ├── __init__.py        # apply_flash_codec / CodecKVCache / JPEGCodec / PNGCodec
│   ├── codecs/jpeg.py     # DCT-II 基底、投影/重构（纯 GEMM，无循环）
│   ├── codecs/png.py      # Anchor + 一阶差分、cumsum 还原、Top-M 阈值滤波
│   ├── cache.py           # CodecKVCache：raw 尾 + 分块归档，全量/增量双接口
│   └── patch.py           # 一行热补丁：Cache 层注入（HF Cache 鸭子类型）
├── tests/                 # jpeg / png / causal 三组测试（直接运行或 pytest）
├── benchmarks/            # 显存压测 + PSNR/余弦质量压测
├── pyproject.toml
└── README.md
```

## 测试

```bash
python tests/test_jpeg.py           # 正交归一、K=r 无损、k=2 余弦>0.99、K 单调
python tests/test_png.py            # 无损 maxdiff<1e-6、方差压缩比打印、可微
python tests/test_causal.py         # 未来扰动零泄露、增量 vs 全量 ≤1e-4
python tests/test_hf_integration.py # transformers 5.x 原生 generate 集成：
                                    # 显式缓存 / 热补丁 / PNG 无损逐 token 一致 /
                                    # 70 token 跨 8+ 块稳定性
```

## 边界与说明（诚实清单）

- 研究原型：`read()` 物化完整重构视图参与注意力，显存收益在**缓存存储层**
  （如上表）；与 flash-attention 类不物化内核结合才能量化端到端收益。
- 全量 `forward(seq)` 预填会把完整块直接压缩，块内位置的注意力会看到块级
  重构（块内近似）；增量 `decode_step` 路径无此问题（当前块恒为 raw）。
- 压缩张量为稠密布局；PNG 的 Top-M 稀疏性尚未接熵编码器。
- 与 [Mini-Architecture](../README.md) 中 v4.2 原型（模型内 KV 复用层的
  DCT/差分读取）同源：那里是"每位置视图"语义，这里是标准 cache 存储语义。

## License

MIT
