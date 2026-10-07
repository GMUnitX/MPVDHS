# -*- coding: utf-8 -*-
"""
分歧分析模块（规则8）
================================================================
把两条路径在本步骤的"最后一层隐藏状态向量序列"用 FastDTW 对齐，
在对齐路径上按 HEAD_SPLIT_RATIO 切分头部/尾部，
头部与头部、尾部与尾部分别计算平均余弦相似度：
  - 头部相似度低于 HEAD_SIM_THRESHOLD                → 创造性分歧
  - 头部达标、尾部相似度低于 TAIL_SIM_THRESHOLD      → 错误性分歧
  - 头尾都达标                                       → 无分歧
"""

import numpy as np

import config

try:
    from fastdtw import fastdtw
except ImportError as exc:  # pragma: no cover
    raise ImportError("缺少 fastdtw 库，请先安装：pip install fastdtw") from exc


def _to_numpy(vectors):
    """list[torch.Tensor(1维)] -> np.ndarray(序列长, 向量维度)"""
    return np.stack([v.detach().float().cpu().numpy() for v in vectors])


def _euclidean(x, y):
    return float(np.linalg.norm(x - y))


def _mean_cosine(seq_a, seq_b, pairs):
    """对 DTW 对齐路径中属于某一区域的所有配对 (i, j) 取余弦相似度的平均。"""
    if not pairs:
        return 1.0  # 该区域为空（序列太短）时不构成分歧信号
    sims = []
    for i, j in pairs:
        va, vb = seq_a[i], seq_b[j]
        norm_a = float(np.linalg.norm(va))
        norm_b = float(np.linalg.norm(vb))
        if norm_a < 1e-12 or norm_b < 1e-12:
            sims.append(0.0)
        else:
            sims.append(float(np.dot(va, vb) / (norm_a * norm_b)))
    return float(np.mean(sims))


def compare_hidden_sequences(hidden_a, hidden_b):
    """
    输入：
        hidden_a / hidden_b —— 两条路径本步骤的隐藏状态向量序列（list，逐 token）。
    返回：
        (head_sim, tail_sim) 头部平均余弦相似度、尾部平均余弦相似度。
    """
    a = _to_numpy(hidden_a)
    b = _to_numpy(hidden_b)
    _, path = fastdtw(a, b, radius=config.FASTDTW_RADIUS, dist=_euclidean)

    head_len = max(1, int(len(path) * config.HEAD_SPLIT_RATIO))
    head_sim = _mean_cosine(a, b, path[:head_len])
    tail_sim = _mean_cosine(a, b, path[head_len:])
    return head_sim, tail_sim
