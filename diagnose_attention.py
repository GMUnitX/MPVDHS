# -*- coding: utf-8 -*-
"""
注意力阈值诊断工具
================================================================
实测"相邻 token 的最后一层多头平均注意力在公共历史部分的余弦相似度"
在正常生成过程中的分布，帮助校准 config.ATTN_SIM_THRESHOLD。

原理与引擎完全一致（规则5）：prev[:-1] 与 cur[:len(prev)-1] 的余弦相似度。
运行：python diagnose_attention.py [提示语] [生成token数]
"""

import sys

import torch

import config
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    prompt_text = sys.argv[1] if len(sys.argv) > 1 else "请用三句话介绍长城。"
    num_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 80

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_PATH,
        dtype=torch.float32,
        attn_implementation=config.ATTN_IMPLEMENTATION,
    )
    model.eval()

    messages = [{"role": "user", "content": prompt_text}]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
    )

    sims = []
    prev_attn = None
    with torch.inference_mode():
        out = model(input_ids=input_ids, use_cache=True)
        past = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        for _ in range(num_tokens):
            out = model(
                input_ids=next_token,
                past_key_values=past,
                use_cache=True,
                output_attentions=True,
            )
            past = out.past_key_values
            attn_vec = out.attentions[-1][0, :, 0, :].mean(dim=0).float()  # 多头平均
            if prev_attn is not None:
                common = prev_attn.shape[0] - 1  # 去掉两 token 自身位置
                a, b = prev_attn[:common], attn_vec[:common]
                sims.append(float((a @ b) / (a.norm() * b.norm() + 1e-12)))
            prev_attn = attn_vec

            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            if int(next_token) in (
                {tokenizer.eos_token_id}
                | (set(model.generation_config.eos_token_id or []) if model.generation_config else set())
            ):
                break

    print(f"提示语: {prompt_text}")
    print(f"生成 token 数: {sum(1 for _ in sims) + 1}")
    print(f"相邻 token 注意力相似度（共 {len(sims)} 个）:")
    line = "  " + " ".join(f"{s:.3f}" for s in sims)
    for i in range(0, len(line), 100):
        print(line[i : i + 100])

    import numpy as np

    arr = np.array(sims)
    print("\n统计:")
    print(f"  min={arr.min():.4f}  mean={arr.mean():.4f}  max={arr.max():.4f}")
    for q in (1, 5, 10, 25, 50):
        print(f"  P{q:02d} = {np.percentile(arr, q):.4f}")
    print("\n建议：ATTN_SIM_THRESHOLD 取 P05 ~ P10 附近（约 5%~10% 的 token 会触发步骤末尾）")


if __name__ == "__main__":
    main()
