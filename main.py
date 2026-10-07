# -*- coding: utf-8 -*-
"""
MPVDHS 命令行对话演示
================================================================
多轮对话：每轮用户输入后，引擎按"步骤"串行推进生成，
每步骤并行 n 条路径并做幻觉抑制裁决（详见 config.py 与 README.md）。
输入 exit / quit 退出。
"""

import config
from mpvdhs.engine import MPVDHSEngine


def main():
    engine = MPVDHSEngine()
    print("=" * 62)
    print("MPVDHS 幻觉抑制推理系统 | 输入 exit 或 quit 退出")
    print("=" * 62)

    messages = []
    while True:
        try:
            user_input = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break

        messages.append({"role": "user", "content": user_input})
        print("AI: ", end="", flush=True)
        result = engine.generate(messages)
        print(result.text)
        print(
            f"[MPVDHS] 停止原因: {result.stop_reason} | 步骤数: {len(result.steps)} "
            f"| 新增 token: {len(result.token_ids)}"
        )
        messages.append({"role": "assistant", "content": result.text})


if __name__ == "__main__":
    main()
