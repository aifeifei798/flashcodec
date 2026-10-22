# AGENTS.md — flashcodec

独立开源项目：把 Transformer 时序 KV Cache 张量当"高维多媒体图像信号"压缩
（jpeg=频域 DCT 有损 / png=时域差分无损）。纯 torch，无 pytest 依赖，
无 CI。改动前先读完本文件。

## 运行约定

- **必须从本目录（`flashcodec/`）内运行**：包与项目目录同名，从父目录
  `import flashcodec` 只会命中命名空间包（没有 `__version__`/导出）——这是
  同名布局的固有陷阱。`pip install -e .` 后任意位置可导入。
- 测试是直跑脚本（`assert` + `__main__` 汇总，函数名兼容 pytest）：
  `python tests/test_jpeg.py`、`python tests/test_png.py`、
  `python tests/test_causal.py`、`python tests/test_hf_integration.py`。
  改 codecs/cache 后四组全跑。
- 基准：`python benchmarks/bench_reconstruct.py`（PSNR/余弦）、
  `python benchmarks/bench_vram.py [--cuda]`（存储占用/耗时）。
- 本目录在宿主仓库里被 gitignore——独立 git 仓库，独立发版。

## transformers 集成（只适配本机 5.20，4.x 明确不支持）

- `FlashCodecLayer(DynamicLayer)` + `FlashCodecCache(Cache,
  layer_class_to_replicate=partial(...))`：5.x 的 Cache 是 layer 容器，
  基类 `update` 懒建层并下发到 layer；**5.x 注意力直接消费 `update` 的
  返回值**（不是读 `key_cache`），所以 layer.update 必须返回重构后的
  完整因果可见 KV。
- `apply_flash_codec` 主钩子是 `model.generate`（5.x generate 内部自建
  DynamicCache，只在入口注入才生效）；forward 包装为辅。
- 生成语义：**cache 长度 = prompt + new_tokens - 1**——末位生成 token 的
  KV 从不被计算（无人 attend），断言别写成总长。
- 集成测试陷阱：随机初始化的微模型会误吐 eos 提前停止
  （`generation_config.eos_token_id = None` 关掉）；lm_head ×20 放大制造
  明确 argmax 间隔，隔离平局抖动、专注检验缓存机制。
- `crop`（投机解码路径）：从裁剪后视图重建归档（原型语义）；
  `offload/prefetch` 为 no-op；beam search（reorder_cache）不支持。

## dtype 契约（Qwen3.5-0.8B 实战踩坑总结，改 codecs 前必读）

- **PNG deltas 恒为 fp32**：bf16 输入的差分在 bf16 里算会掉 ~0.4% 精度，
  cumsum 重建不再逐位精确（bf16 差分在 fp32 中精确可表示，anchor+ΣΔ
  可逐位还原原值）——这是"无损 → 生成 token 与基线逐位一致"的根据。
  bf16 部署下归档约 1.9× dense（带宽换精确）。
- **JPEG coeffs 存输入 dtype**（fp32 计算后 cast）：有损档无需更高精度，
  且存储比不虚高（k=2 实测 ~27% raw）。
- **`_CodecStream.read()` 统一还原流输入 dtype**：SDPA 要求 q/k/v 同
  dtype，fp32 视图喂给 bf16 模型会直接崩。
- 混合架构（Qwen3.5：18 linear_attention + 6 full_attention）：必须用
  `FlashCodecCache.from_config(model.config)` 按 layer_types 只给
  full_attention 层挂 codec，其余层保留原生动态层（递归状态等接口
  不受影响）；`apply_flash_codec` 已自动走 from_config。
- 实战脚本：`benchmarks/run_qwen_demo.py`（PNG 无损逐位一致断言 +
  JPEG 存储比/流畅度报告 + 混合层审计）。

## 张量契约（改动任何签名前必读）

- KV 布局 `(B, H, S, D)`；**压缩/滤波操作轴恒为 dim=-2（seq 轴）**。
- codecs 内只允许批量矩阵乘/逐元素算子，**禁止显式 Python for 循环**；
  尾部不足块用零填充（`_pad_to_blocks`），`out_len` 裁回。
- codecs 统一 fp32 计算（`dct_compress` 等内部 `.float()`）；keep=r 的
  DCT 与 keep_ratio=1 的 PNG 是"无损"基线（fp 往返 ~1e-6/1e-7 级，
  断言容差见测试，别收紧到逐位）。

## 因果安全不变量（破坏即 bug）

1. **只归档完整块**；未满块永远留 `raw_buffer`（原始精度）——增量解码
   任意时刻的未来扰动不得改写已归档张量（`test_causal` 逐位验证）。
2. 全量 `forward(seq)` 与逐步 `decode_step` 两条填充路径产生的归档与
   视图必须**逐位一致**（归档按块分片，解压逐片进行；不要改成"大 chunk
   一次解压 vs 逐块解压"的不对称实现）。
3. `mode="png", keep_ratio=1.0` 无损档位下，逐步解码注意力必须与全量
   前向对齐 ≤1e-4（测试的正确性锚点）。

## 设计决策（别"顺手重构"掉）

- **HF 热补丁在 Cache 层而非注意力内核**：标准 HF 模型的 KV 写入/读取
  统一收敛在 Cache 接口（`update`/`key_cache`），`CodecKVCache` 鸭子类型
  注入即可，比逐模型手术（GQA/MQA/RoPE 变体）鲁棒得多。
- **PNG 稠密存储如实计 1.0**：锚点+差分与原始块等大；省的是信息熵
 （方差压缩 ~30×），压缩空间留给熵编码器——`storage_ratio()` 不许"四舍
  五入"成 (r-1)/r。
- 全量 `forward(seq)` 预填会把完整块直接压缩，块内位置注意力看到块级
  重构（块内近似）；增量路径无此问题。这是已文档化的语义，不是 bug。
- 归档数据结构：jpeg 存 coeffs `(B,H,n,k,D)` 单张量；png 存
  `(deltas, anchor)` 元组——`cache.py` 与测试里都有 isinstance 分支，
  新增模式时保持该约定。

## 发版

- `pyproject.toml`（setuptools，MIT，torch>=2.0），版本号在
  `pyproject.toml` 与 `flashcodec/__init__.py::__version__` 双写，改一处
  必须同步另一处。
- 发 PyPI 前必跑三组测试 + 两个基准（README 里的实测表要随之更新——
  README 的数字都是真实跑出来的，不许留占位或过期数据）。
