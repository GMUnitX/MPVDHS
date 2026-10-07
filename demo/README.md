# demo

独立演示目录,不改动项目原有文件。

## streaming_chat.py — 交互式流式输出对话 demo

在项目根目录运行:

```bash
python demo/streaming_chat.py            # 完整模式
python demo/streaming_chat.py --concise  # 简洁模式
```

完整模式下每完成一个"步骤"(规则4~9 的一轮多路径并行解码 + 裁决)
就立即打印该步骤的文本与诊断面板(各路径状态、分歧裁决、
head/tail 相似度)。

`--concise` 简洁模式只输出模型回复本身,不显示任何过程信息;
`<self-check>` 标记按原样保留在输出中。

demo 运行时只在内存中把 `config.VERBOSE` 置为 `False`(避免与
诊断面板重复打印),不写回任何文件。
