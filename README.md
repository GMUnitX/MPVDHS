# MPVDHS —— Multi-Path Parallel Vector-Divergence Hallucination Suppression（多路径并行向量分歧检测幻觉抑制）

基于 HuggingFace transformers 实现的大语言模型推理系统：每次生成被切成若干**步骤**，
每个步骤从当前 KV 缓存并行分叉出 n 条采样路径，逐 token 监测注意力 / 置信度 / 隐藏状态，
步骤末尾用 FastDTW 头尾相似度判定"创造性分歧 / 错误性分歧 / 无分歧"，
只有唯一胜者的 KV 缓存会进入下一步骤，从而抑制幻觉的传播与放大。

## 目录结构

```
MPVDHS/
├── config.py            # 全部可调参数（分块注释，见下文"参数说明"）
├── main.py              # 命令行多轮对话入口
├── smoke_test.py        # 冒烟测试（小参数快速验证全链路）
├── diagnose_attention.py # 注意力相似度分布诊断（校准 ATTN_SIM_THRESHOLD 用）
├── requirements.txt
├── mpvdhs/
│   ├── engine.py        # 核心引擎：加载模型、多路径解码、步骤裁决
│   └── divergence.py    # FastDTW 对齐 + 头/尾余弦相似度（规则8）
├── models/              # 本地模型（如Qwen2.5-0.5B-Instruct）
```

## 安装与运行

```bash
pip install -r requirements.txt

python smoke_test.py          # 先跑冒烟测试（参数已收紧，CPU 上几十秒）
python diagnose_attention.py  # 实测当前模型的注意力相似度分布，校准 ATTN_SIM_THRESHOLD
python main.py                # 进入交互式多轮对话
```

## 工作流程（对应需求规则 1~9）

1. **加载模型**：按 `config.py` 的 `MODEL_PATH` 加载；注意力实现强制 `eager`
   （SDPA/FlashAttention 拿不到逐 token 注意力权重）。
2. **自动选设备**：`DEVICE_MODE="auto"` 时有 CUDA 用 GPU（fp16），否则 CPU（fp32）。
3. **参数集中**：所有参数在 `config.py`，按 9 个功能分块，逐项注释。
4. **多路径并行**：每个步骤把当前 KV 缓存复制 n 份组成一个 batch，
   n 条路径**相同采样参数**、各自独立采样；第一步用对话模板
   `apply_chat_template(add_generation_prompt=True)` 预填充，后续步骤直接复用胜者缓存。
5. **注意力监测**：每生成一个 token，取最后一层注意力、按注意力头平均成一条向量，
   与上一个 token 的向量计算**公共历史部分**（双方都去掉自己所在位置）的余弦相似度；
   低于 `ATTN_SIM_THRESHOLD` 或采样到 EOS → 到达"该步骤末尾"，暂停该路径并保存其
   KV 缓存，标记"步骤完成"或"EOS"。注意力向量只滑动保留最近一个（等价于"用过两次即删"）。
6. **置信度监测**：每个 token 的置信度 = 采样时模型原始分布（不乘温度）的 top-1 概率；
   低于 `CONFIDENCE_THRESHOLD` → 该路径"提前终止"。正常到达步骤末尾的路径，
   取本步骤所有 token 置信度的**几何平均**作为路径置信度。
7. **隐藏状态序列**：每个 token 的最后一层隐藏向量逐 token 追加；提前终止的路径整条删除。
8. **分歧裁决**：所有路径结束后，未提前终止者按几何平均置信度排序取前二，
   两条隐藏状态序列 FastDTW 对齐 → 对齐路径前 `HEAD_SPLIT_RATIO`（默认30%）为头部：
   - 头部相似度 < `HEAD_SIM_THRESHOLD` → **创造性分歧**：保留胜者缓存，进入下一步骤；
   - 头部达标但尾部 < `TAIL_SIM_THRESHOLD` → **错误性分歧**：保留胜者缓存并强制拼接
     `<self-check>`，进入下一步骤；
   - 头尾都达标 → **无分歧**：同创造性分歧处理。
   每步结束后清除所有路径的注意力记录、置信度序列、隐藏状态序列与标记。
9. **规则9 兜底**：全部路径提前终止（或按用户决定，仅剩 1 条幸存）时，
   保留几何平均置信度最高路径的 KV 缓存 + 强制拼接 `<self-check>`，进入下一步骤。

整段回复在"胜者以 EOS 结束且未拼接 `<self-check>`"时结束；
最终文本 = 每个步骤胜者新增 token 依次拼接。

### 已确认的设计决定（与原始规则的差异点）

- 规则6 说提前终止"不留 KV 缓存"，规则9 又要保留其中最优者的缓存 →
  **提前终止路径的缓存与几何平均置信度在步骤内暂留**（不参与排序与 DTW），
  步骤裁决后随其他路径缓存一并清除。
- 仅剩 1 条路径幸存（无法做两两 DTW）→ **按规则9处理**：保留它并追加 `<self-check>`。

## 参数说明（config.py 分块）

| 分块 | 关键参数 | 说明 |
|---|---|---|
| 1 模型与设备 | `MODEL_PATH` `DEVICE_MODE` `GPU_DTYPE` `CPU_DTYPE` `ATTN_IMPLEMENTATION` | 模型路径、设备与精度；`ATTN_IMPLEMENTATION` 必须保持 `"eager"` |
| 2 并行路径 | `NUM_PATHS` | 每步骤并行路径数 n |
| 3 采样 | `TEMPERATURE` `TOP_P` `TOP_K` `SEED` | n 条路径共用；`TEMPERATURE` 必须 > 0，否则路径全部相同、分歧检测失效 |
| 4 上限 | `MAX_STEPS` `MAX_STEP_TOKENS` `MAX_TOTAL_NEW_TOKENS` | 防跑飞的安全阀；步内达到上限强制按"步骤完成"结束 |
| 5 注意力 | `ATTN_SIM_THRESHOLD` `ATTN_MIN_COMMON_LEN` | 相似度低于阈值即到达步骤末尾；建议 0.85~0.95 试探 |
| 6 置信度 | `CONFIDENCE_THRESHOLD` | top-1 概率低于阈值即提前终止；建议 0.05~0.30 |
| 7 分歧 | `HEAD_SPLIT_RATIO` `HEAD_SIM_THRESHOLD` `TAIL_SIM_THRESHOLD` `FASTDTW_RADIUS` | 头尾切分比例与两条阈值 |
| 8 self-check | `SELF_CHECK_TOKEN` `SELF_CHECK_AS_SPECIAL` `SELF_CHECK_SHOW_IN_OUTPUT` | 默认按普通文本分词拼接（不改动模型）；计划微调模型识别该标记时再改为 True |
| 9 日志 | `VERBOSE` | 每步骤打印各路径状态与裁决详情 |

## 编程接口

```python
from mpvdhs import MPVDHSEngine

engine = MPVDHSEngine()
messages = [{"role": "user", "content": "你好"}]
result = engine.generate(messages)   # messages 遵循标准对话格式，自动套模板
print(result.text)                   # 最终回复
print(result.stop_reason)            # eos / max_steps / max_total_tokens
print(result.steps)                  # 每个步骤的诊断信息
```

## 实现注意事项

- **内存**：每步骤同时持有 n 条路径的 KV 缓存（批量前向），步骤结束后只保留胜者。
  已结束路径用占位 token 凑批继续前向，其缓存增长部分在步骤末尾裁剪。
- **CPU 速度**：0.5B 模型 + n=4 时单步解码较慢属于正常；可调小 `NUM_PATHS`、
  `MAX_STEP_TOKENS`、`MAX_TOTAL_NEW_TOKENS` 加速。
- **transformers 兼容**：KV 缓存操作同时兼容旧版（`key_cache/value_cache`）与
  新版 4.56+（`layers[i].keys/values`）结构。
- **`<self-check>`**：模型原本不认识该标记。默认按普通文本分词拼接，仅作为上下文信号；
  若想让模型真正学会响应它，需自行微调，并把 `SELF_CHECK_AS_SPECIAL` 改为 `True`。
