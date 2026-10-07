# -*- coding: utf-8 -*-
"""
MPVDHS 交互式流式输出对话 demo（一个步骤一输出）
================================================================
不修改 mpvdhs 包内任何文件：通过子类复用引擎内部方法，
把 generate() 的外层步骤循环重写为生成器，每完成一个"步骤"
（规则4~9 的一轮多路径并行解码 + 裁决）就立即输出该步骤的文本
与本步骤的诊断信息，而不是等整段回复生成完。

运行方式（在项目根目录）:
    python demo/streaming_chat.py            # 完整模式：显示步骤诊断面板
    python demo/streaming_chat.py --concise  # 简洁模式：只输出模型回复（含 <self-check>）
输入 exit / quit 退出。
"""

import argparse
import os
import sys

# 保证无论从哪个工作目录启动，都能 import 到项目根目录的 config 与 mpvdhs
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from mpvdhs.engine import MPVDHSEngine, GenerationResult


class StreamingEngine(MPVDHSEngine):
    """在 MPVDHSEngine 基础上提供按步骤流式输出的生成器接口。

    generate_stream() 逐段 yield (step_text, step_info)：
      step_text  —— 本步骤计入最终回复的文本
      step_info  —— 本步骤的诊断信息 dict（路径状态、裁决、相似度等）
    最终 return 一个 GenerationResult（与基类 generate() 返回结构一致）。
    """

    def generate_stream(self, messages, show_special=False):
        if not messages:
            raise ValueError("messages 不能为空")

        prompt_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )
        if torch_is_tensor(prompt_ids):
            prompt_ids = prompt_ids.reshape(-1).tolist()

        cache, start_logits = self._prefill(prompt_ids)

        output_ids, steps_info = [], []
        stop_reason = "max_steps"
        for step_idx in range(1, config.MAX_STEPS + 1):
            if len(output_ids) >= config.MAX_TOTAL_NEW_TOKENS:
                stop_reason = "max_total_tokens"
                break
            result = self._run_step(cache, start_logits, step_idx)
            output_ids.extend(result.output_token_ids)
            cache = result.cache
            start_logits = result.next_logits
            steps_info.append(result.info)

            step_text = self.tokenizer.decode(
                result.output_token_ids, skip_special_tokens=not show_special
            )
            yield step_text, result.info

            if result.end_generation:
                stop_reason = "eos"
                break

        yield None, GenerationResult(
            text=self.tokenizer.decode(
                output_ids, skip_special_tokens=not show_special
            ),
            token_ids=output_ids,
            stop_reason=stop_reason,
            steps=steps_info,
        )


def torch_is_tensor(obj):
    import torch
    return torch.is_tensor(obj)


def print_step_panel(step_idx, step_text, info):
    """把一个步骤的文本与诊断信息作为一个面板打印。"""
    sim = ""
    if info.get("head_sim") is not None:
        sim = f" | head_sim={info['head_sim']} tail_sim={info['tail_sim']}"
    suffix = " + <self-check>" if info["self_check"] else ""
    print(f"\n┌── 步骤 {step_idx} | {info['verdict']}{sim} "
          f"→ 保留路径{info['kept_path']}{suffix} ──")
    for p in info["paths"]:
        print(f"│ 路径{p['index']}: {p['status']}({p['end_reason'] or '-'}) "
              f"tokens={p['num_tokens']} geoConf={p['geomean_confidence']}")
    print(f"└─ 输出: {step_text}")


def main():
    parser = argparse.ArgumentParser(description="MPVDHS 流式对话 demo")
    parser.add_argument(
        "--concise", action="store_true",
        help="简洁模式：只输出模型回复（含 <self-check>），不显示步骤诊断过程",
    )
    args = parser.parse_args()
    concise = args.concise

    # demo 自己按步骤打印诊断信息，关掉引擎内部的 VERBOSE 打印避免重复
    config.VERBOSE = False

    print("[MPVDHS] 正在加载模型（首次加载需要一些时间）...")
    engine = StreamingEngine()
    print("=" * 62)
    if concise:
        print("MPVDHS 流式对话 demo（简洁模式）| 输入 exit / quit 退出")
    else:
        print("MPVDHS 流式对话 demo（一个步骤一输出）| 输入 exit / quit 退出")
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
        if concise:
            print("AI: ", end="", flush=True)
        else:
            print("AI:", flush=True)
        try:
            gen = engine.generate_stream(messages, show_special=concise)
            while True:
                step_text, info = next(gen)
                if step_text is None:  # 结束标记：info 为 GenerationResult
                    final = info
                    break
                if concise:
                    # 只追加本步骤文本，<self-check> 等特殊 token 原样保留
                    print(step_text, end="", flush=True)
                else:
                    print_step_panel(info["step"], step_text, info)
        except (KeyboardInterrupt, StopIteration):
            print("\n[MPVDHS] 本轮生成被中断")
            continue

        if concise:
            print(flush=True)  # 换行结束本轮回复，不打印任何诊断信息
        else:
            print(
                f"\n[MPVDHS] 停止原因: {final.stop_reason} | 步骤数: {len(final.steps)} "
                f"| 新增 token: {len(final.token_ids)}"
            )
        messages.append({"role": "assistant", "content": final.text})


if __name__ == "__main__":
    main()
