# flashcodec

> 把 Transformer 的时序 KV Cache 张量当作**高维多媒体图像信号**来压缩。

[English](README.md) | [简体中文](README.zh-CN.md)

传统 LLM 研究把隐空间张量视为孤立标量因子；flashcodec 认为 `(B, H, S, D)`
布局里的 **seq 轴是一幅高维图像的行/列**——相邻 token 隐向量自相关、能量集中
在低频、差分后方差收缩（具体程度取决于模型/层/张量：真实 KV 实测统计与逐层
注意事项见「基准」一节）。于是数字多媒体领域的两大经典压缩理论可以
直接迁移：

| 模式 | 灵感 | 变换 | 性质 | 存储 |
|---|---|---|---|---|
| `mode="jpeg"` | JPEG | 分块正交 1D DCT-II，保留最低 K 个低频系数 | **有损频域** | `K/r`（4×~8× 显存节省） |
| `mode="png"` | PNG | 块首锚点（I-Token）+ 一阶差分 Δ，Top-M 裁剪 | **无损/近无损时域** | 稠密等大；残差稀疏性留给熵编码 |

`quant_bits=8`（仅 jpeg）把系数存成 int8 + 每个（块, 频率）一个 scale——在
`K/r` 之上再省真实字节：**k=2 时只占原始 KV 的 6.6%（15×），PSNR 46.9 dB**，
而 fp32 系数是 25% / 47.2 dB。

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
                    │        │ 满 r 个 token，批量归档（一次 GEMM │
                    │        ▼  原地切片写入）                    │
                    │  compressed_storage（倍增预分配）           │
                    │   jpeg: 系数 (B,H,capacity,K,D)             │
                    │   png:  deltas (B,H,capacity,r-1,D)         │
                    │        + anchors (B,H,capacity,D)           │
                    └───────────────┬────────────────────────────┘
                                    │ read()：有效区解码 + raw 尾
                                    ▼
                            重构 KV 视图 (B,H,S,D)
```

归档存放在**单块几何倍增缓冲区**里（std::vector 式：初始 16 块，装满翻倍）。
每满块只做一次原地切片写入（无反复重新分配）；`read()` 只解码**新增归档块**
（每个新块一次批量 GEMM / cumsum，落在追加式重构缓存里），不再每个 decode step
重解全部历史。若改成逐块碎片归档 + 逐归档循环解压，S=4096 的逐步解码
read 会因 kernel launch 开销慢约 170×（实测 31ms → 0.18ms）。

## 因果安全设计

只归档**完整块**；当前未满块永远留在 `raw_buffer`（原始精度）。增量解码时
任意时刻的未来扰动都不会改写已归档张量（测试逐位验证）。`mode="png",
keep_ratio=1.0` 的无损档位下，逐步解码注意力与全量前向注意力对齐 ≤1e-4。

## 安装

```bash
pip install -e ".[hf]"  # 源码安装：torch>=2.0（HF 后端另需 transformers>=5.20）
```

PyPI 分发包名：`flashcodec-kv`，导入名仍是 `flashcodec`（PyPI 上的
`flashcodec` 是无关占位包，别装错）。

## 快速上手

```python
import torch
from flashcodec import FlashCodecCache, apply_flash_codec

# 1) 原生：transformers 5.x 一等公民 Cache 后端（任意 CausalLM）
cache = FlashCodecCache(mode="jpeg", block_size=8, keep_freqs=2)  # 4×
#    int8 系数：同样形状，归档再小 15×（quant_bits=8）
cache = FlashCodecCache(mode="jpeg", block_size=8, keep_freqs=2,
                        quant_bits=8)                              # 15×
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
| `png keep=1.0` | ~1.8×（fp32 差分换精确）| **生成 token 与未打补丁模型逐位一致** | 100% 保真基线 |
| `jpeg k=8 int8` | **55.5%**（约 1.8×）| 文本流畅，**token 一致率 100%** | 安全有损档 |
| `jpeg k=2` | **32.3%**（约 3.1×）| 文本流畅，token 一致率 38.1% | 有损档 |
| `jpeg k=4` | 54.8%（约 1.8×）| 文本流畅，token 一致率 49.2% | 有损档 |
| `jpeg k=2 int8` | **21.1%**（约 4.7×）| 文本流畅，token 一致率 38.1% | 有损 + int8 |

混合层审计：`codec_layers=6/24`（仅全注意力层），零警告。复跑：
`python benchmarks/run_qwen_demo.py`（离线加载；支持 `FLASHCODEC_MODEL` /
`FLASHCODEC_NEW_TOKENS` 环境变量）。

dtype 契约（详见 `AGENTS.md`）：PNG 差分恒为 fp32（bf16 差分会破坏无损
档承诺的逐位精确性）；JPEG 系数与 `read()` 视图保持流输入 dtype
（SDPA 要求 q/k/v 同 dtype）。

## Demo（Qwen3.5-0.8B 左右对比 Chat）

![flashcodec Gradio 左右对比 demo](images/demo.png)

同一 prompt 两栏并排：左侧原生 KV Cache，右侧 flashcodec（档位可切：
png 无损 / jpeg k=1,2,4），底部柱状图对比存储与耗时的节约，并附 token
一致率表。

```bash
pip install -e ".[demo]"   # gradio + matplotlib
python demo/qwen_gradio_chat.py    # Gradio 左右对比 + 节约柱状图
```

## 基准（实测，B=2 H=8 S=2048 D=64，随机游走自相关信号，fp32）

### 重构质量（`benchmarks/bench_reconstruct.py`）

| 配置 | PSNR (dB) | cosine | 存储比 |
|---|---|---|---|
| jpeg r=8 k=1（8×） | 42.95 | 0.99943 | 12.5% |
| jpeg r=8 k=2（4×） | 47.22 | 0.99985 | 25.0% |
| jpeg r=8 k=4（2×） | 51.66 | 1.00001 | 50.0% |
| jpeg r=8 k=2 int8 | 46.92 | 0.99983 | 6.6% |
| jpeg r=8 k=4 int8 | 50.89 | 0.99999 | 13.3% |
| png r=8 keep=0.5 | 50.11 | 0.99997 | 100%（稀疏留熵编码）|
| png r=8 keep=1.0 | ~196（fp 舍入级）| 1.00000 | 100% |

### 真实 KV 统计（真实模型，512 token）

`bench_reconstruct` 用的是合成随机游走；真实 cache 没那么光滑。在真实
post-RoPE KV 上实测（沿 seq 轴的 `Var(Δ)/Var(X)`，以及块长 8 时最低 k=2 个
DCT 系数保留的块能量占比）：

| 模型 / 层 | 张量 | Var(Δ)/Var(X) | k=2 保留能量 |
|---|---|---|---|
| Qwen3-0.6B L0 | K | 0.16 | 0.94 |
| Qwen3-0.6B L0 | V | 1.87 | 0.31 |
| Qwen3-0.6B L27 | V | 1.02 | 0.59 |
| Qwen3.5-0.8B L3 | K | 0.72 | 0.71 |
| Qwen3.5-0.8B L23 | V | 0.09 | 0.97 |

V 经常是**反**相关的（比值 > 1），所以「一阶差分让方差塌缩」是光滑合成信号
的性质，不是每一层真实 KV 的性质。实际结论：系数截断（`k < r`）必须按层决策
——稠密模型（28 层全压）会误差累积到崩，那里请用 `k = r` + `quant_bits=8`。

### 困惑度（`benchmarks/bench_ppl.py`；16 × 512 token，bf16，本地语料）

| 档位 | Qwen3.5-0.8B（压 6/24 层） | Qwen3-0.6B（压 28/28 层） | 存储 |
|---|---|---|---|
| baseline | 22.4803 | 16.7931 | 100% |
| png keep=1.0 | 22.4803 (+0.000) | 16.7931 (+0.000) | 188% |
| jpeg k=8（无损基） | 22.4877 (+0.007) | 16.8058 (+0.013) | 100% |
| **jpeg k=8 int8** | **22.4991 (+0.019)** | **17.1348 (+0.34)** | **~52%** |
| jpeg k=4 | 22.3010 (−0.18) | 470.85 (+454) | 50% |
| jpeg k=2 | 24.0566 (+1.58) | 6485.5 (+6469) | 25% |
| jpeg k=2 int8 | 24.0590 (+1.58) | 8118.5 (+8102) | 13% |

混合架构结果好是因为只有 6/24 层挂了 codec（18 层线性注意力保留原生状态）；
稠密模型那两列说明了为什么按层截断不能盲用。

### 缓存占用与耗时（`benchmarks/bench_vram.py`，CPU）

| seq | 模式 | storage | 原始比 | fill | read |
|---|---|---|---|---|---|
| 4096 | jpeg k=2 | 4.0MB | 0.25 | 7.4ms | 0.1ms |
| 16384 | jpeg k=2 | 16.0MB | 0.25 | 31.6ms | 0.3ms |
| 65536 | jpeg k=2 | 64.0MB | 0.25 | 125.0ms | 3.2ms |
| 4096 | jpeg k=2 int8 | 1.1MB | 0.07 | 11.2ms | 0.1ms |
| 16384 | jpeg k=2 int8 | 4.3MB | 0.07 | 44.0ms | 0.4ms |
| 65536 | jpeg k=2 int8 | 17.0MB | 0.07 | 180.5ms | 3.2ms |
| 65536 | png keep=1.0 | 256.0MB | 1.00 | 224.1ms | 3.2ms |

CPU 计时逐次波动约 ±20%（取 3 次最小）；storage/ratio 两列是精确值。
`read` 为「物化视图并真实读取（`.sum()`）」的耗时，即消费端成本而非仅 kernel
启动。解码按归档块记忆化，`read` 只在**新增块**上做一次解码，不再每个 decode
step 重解全部历史（此前最后两行为 38.7ms / 71.9ms）。

PNG 模式不省稠密存储（无损基线的宿命）——它的价值在 30× 方差压缩 + Top-M
近零稀疏残差，是熵编码器的理想输入（本库不内置熵编码）。

## 目录结构

```
flashcodec/
├── flashcodec/
│   ├── __init__.py        # apply_flash_codec / CodecKVCache / JPEGCodec / PNGCodec
│   ├── codecs/jpeg.py     # DCT-II 基底、投影/重构（纯 GEMM，无循环）
│   ├── codecs/png.py      # Anchor + 一阶差分、cumsum 还原、Top-M 阈值滤波
│   ├── cache.py           # CodecKVCache：raw 尾 + 倍增预分配归档，全量/增量双接口
│   └── patch.py           # 一行热补丁：Cache 层注入（HF Cache 鸭子类型）
├── tests/                 # jpeg / png / causal 三组测试（直接运行或 pytest）
├── benchmarks/            # 显存/存储 + PSNR/余弦 + 困惑度压测
├── demo/                  # Gradio 左右对比 chat
├── images/                # README 截图
├── .github/workflows/     # CI：CPU 测试 + transformers 集成
├── CHANGELOG.md
├── pyproject.toml
└── README.md
```

## 测试

```bash
python tests/test_jpeg.py           # 正交归一、K=r 无损、k=2 余弦>0.99、K 单调
python tests/test_png.py            # 无损 maxdiff<1e-6、方差压缩比打印、可微
python tests/test_causal.py         # 未来扰动零泄露、增量 vs 全量 ≤1e-4、
                                    # 记忆化 read 与全量解码逐位一致、crop 精确且可重复、
                                    # beam search 批维重排
python tests/test_hf_integration.py # transformers 5.x 原生 generate 集成：
                                    # 显式缓存 / 热补丁 / PNG 无损逐 token 一致 /
                                    # 70 token 跨 8+ 块稳定性 / beam search 与 crop
                                    # 对未打补丁模型的精确性
```

## 边界与说明（诚实清单）

- 研究原型：`read()` 物化完整重构视图参与注意力，显存收益在**缓存存储层**
  （如上表）；与 flash-attention 类不物化内核结合才能量化端到端收益。该视图
  被记忆化并跨 decode step 复用，因此稳态显存 = 归档 + 一份原始 KV 大小的视图。
- 全量 `forward(seq)` 预填会把完整块直接压缩，块内位置的注意力会看到块级
  重构（块内近似）；增量 `decode_step` 路径无此问题（当前块恒为 raw）。
- 压缩张量为稠密布局；PNG 的 Top-M 稀疏性尚未接熵编码器（实测只做稀疏打包
  基本是白忙：fp32 差分本身占大头，bf16 输入下甚至可能比原始更大——得先做
  差分量化）。
- 系数截断（`k < r`）必须按层决策：稠密模型每层都压会误差累积（实测
  Qwen3-0.6B 困惑度 16.8 → 6486 @ k=2）。这种模型请用 `k = r` +
  `quant_bits=8`（约 52% 存储，+2% ppl）；激进压缩比留给混合架构（只有
  full_attention 层挂 codec 的情形）。
- 与 [Mini-Architecture](https://github.com/aifeifei798/Mini-Architecture)
  中 v4.2 原型（模型内 KV 复用层的 DCT/差分读取）同源：那里是"每位置视图"
  语义，这里是标准 cache 存储语义。

## License

MIT
