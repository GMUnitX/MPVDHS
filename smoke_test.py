# -*- coding: utf-8 -*-
"""
冒烟测试：加载真实本地模型，跑一个小生成，验证全链路
（预填充 -> 多路径并行 -> 注意力/置信度/隐藏状态监测 -> 分歧裁决 -> <self-check> -> 输出）。
运行：python smoke_test.py
"""

import config

# 收紧参数，让测试在 CPU 上快速跑完（不影响 config.py 的正式配置）
config.NUM_PATHS = 3
config.MAX_STEPS = 2
config.MAX_STEP_TOKENS = 32
config.MAX_TOTAL_NEW_TOKENS = 96
config.VERBOSE = True

from mpvdhs.engine import MPVDHSEngine  # noqa: E402


def main():
    engine = MPVDHSEngine()
    messages = [{"role": "user", "content": "请用三句话介绍长城。"}]
    result = engine.generate(messages)

    print("\n===== 最终输出 =====")
    print(result.text)
    print(f"停止原因: {result.stop_reason} | 步骤数: {len(result.steps)} | 新增 token: {len(result.token_ids)}")

    assert isinstance(result.text, str), "输出必须是字符串"
    print("\n冒烟测试通过 ✓")


if __name__ == "__main__":
    main()
