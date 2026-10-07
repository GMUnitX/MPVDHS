# -*- coding: utf-8 -*-
"""MPVDHS：多路径并行解码 + 动态幻觉抑制推理系统"""
from .engine import MPVDHSEngine, GenerationResult, StepResult
from .divergence import compare_hidden_sequences

__all__ = ["MPVDHSEngine", "GenerationResult", "StepResult", "compare_hidden_sequences"]
