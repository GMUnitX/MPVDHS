# -*- coding: utf-8 -*-
"""
MPVDHS 核心引擎
================================================================
多路径并行解码 + 动态幻觉抑制推理系统。

与需求的对应关系：
  规则1/2  加载模型、自动选设备              -> MPVDHSEngine.__init__
  规则4    每步骤并行 n 条路径、复用上一步KV -> _run_step / replicate_cache
  规则5    注意力监测、步骤末尾、EOS         -> 解码循环内 ATTENTION 检查
  规则6    置信度监测、提前终止、几何平均     -> sample_batch / PathState.finalize
  规则7    逐 token 隐藏状态序列             -> PathState.hidden_vectors
  规则8    前二名 FastDTW 头尾分歧判定       -> _resolve_step
  规则9    全部提前终止 / 仅剩1条时兜底      -> _resolve_step
"""

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

import config
from .divergence import compare_hidden_sequences

# ---------------------------------------------------------------------------
# 路径状态与结束原因（对应规则5/6 的标记）
# ---------------------------------------------------------------------------
STATUS_RUNNING = "运行中"
STATUS_STEP_DONE = "步骤完成"   # 规则5：注意力相似度低于阈值，或达到步内 token 上限
STATUS_EOS = "EOS"              # 规则5：采样到结束符
STATUS_EARLY = "提前终止"        # 规则6：置信度低于阈值

END_ATTENTION = "注意力相似度低于阈值"
END_MAX_TOKENS = "步内token上限"
END_EOS = "EOS"
END_CONFIDENCE = "置信度低于阈值"


# ---------------------------------------------------------------------------
# KV 缓存辅助函数（兼容 transformers 新旧版本的 DynamicCache 结构）
# ---------------------------------------------------------------------------
def _iter_cache_layers(cache):
    """迭代 KV 缓存每层的 (keys, values)，形状均为 (batch, kv_heads, seq_len, head_dim)。"""
    key_cache = getattr(cache, "key_cache", None)  # 旧版 transformers
    if key_cache:
        return list(zip(key_cache, cache.value_cache))
    layers = getattr(cache, "layers", None)        # 新版 transformers (>=4.56)
    if layers:
        result = []
        for layer in layers:
            k, v = getattr(layer, "keys", None), getattr(layer, "values", None)
            if not torch.is_tensor(k) or not torch.is_tensor(v):
                raise RuntimeError(f"无法读取 KV 缓存层内容: {type(layer)}")
            result.append((k, v))
        return result
    raise RuntimeError(f"不支持的 KV 缓存类型: {type(cache)}")


def get_cache_seq_len(cache):
    """缓存中已存有多少个 token 的 KV。"""
    keys, _ = _iter_cache_layers(cache)[0]
    return int(keys.shape[2])


def replicate_cache(cache, n):
    """把 batch=1 的 KV 缓存复制 n 份 -> batch=n 的新缓存（规则4：n 条路径同起点）。"""
    new_cache = DynamicCache()
    for layer_idx, (k, v) in enumerate(_iter_cache_layers(cache)):
        new_cache.update(k.repeat(n, 1, 1, 1), v.repeat(n, 1, 1, 1), layer_idx)
    return new_cache


def slice_path_cache(cache, path_index, seq_len):
    """从 batch=n 的缓存中取出第 path_index 条路径，并截断到 seq_len（去掉
    路径结束后为凑批而喂入的占位 token），得到 batch=1 的新缓存。"""
    new_cache = DynamicCache()
    for layer_idx, (k, v) in enumerate(_iter_cache_layers(cache)):
        new_cache.update(
            k[path_index : path_index + 1, :, :seq_len, :].contiguous(),
            v[path_index : path_index + 1, :, :seq_len, :].contiguous(),
            layer_idx,
        )
    return new_cache


# ---------------------------------------------------------------------------
# 采样（规则4：n 条路径相同参数；规则6：置信度=原始分布 top-1 概率）
# ---------------------------------------------------------------------------
def sample_batch(logits):
    """
    对 batch 内每一行独立采样一次。
    输入: logits (n, vocab) —— 本步骤所有路径的下一 token 分布。
    返回: (tokens (n,), top1_probs (n,))
          top1_probs 按"不乘温度的原始分布"计算，作为该 token 的置信度。
    """
    raw_probs = F.softmax(logits, dim=-1)
    top1_probs = raw_probs.max(dim=-1).values

    temperature = config.TEMPERATURE
    if temperature is None or temperature <= 1e-6:  # 贪心（不推荐，见 config 注释）
        return logits.argmax(dim=-1), top1_probs

    probs = F.softmax(logits / temperature, dim=-1)

    top_k = config.TOP_K
    if top_k and top_k > 0:
        k = min(int(top_k), probs.size(-1))
        kth_value = torch.topk(probs, k, dim=-1).values[:, -1:]  # 每行第 k 大的概率值
        probs = probs.masked_fill(probs < kth_value, 0.0)

    top_p = config.TOP_P
    if top_p is not None and top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cumulative = sorted_probs.cumsum(dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()  # 右移一位：保证至少保留 1 个 token
        remove[..., 0] = False
        remove = remove.scatter(dim=-1, index=sorted_idx, src=remove)
        probs = probs.masked_fill(remove, 0.0)

    probs = probs / probs.sum(dim=-1, keepdim=True)
    tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
    return tokens, top1_probs


def cosine_similarity(a, b):
    """两个 1 维向量的余弦相似度（零向量按 0 处理）。"""
    a, b = a.float(), b.float()
    denom = a.norm() * b.norm()
    if denom < 1e-12:
        return 0.0
    return float((a @ b) / denom)


# ---------------------------------------------------------------------------
# 路径状态（规则5/6/7 的全部记录）
# ---------------------------------------------------------------------------
@dataclass
class PathState:
    index: int                                        # 在批量中的位置
    status: str = STATUS_RUNNING
    end_reason: str = ""
    token_ids: list = field(default_factory=list)     # 本步骤生成的 token（EOS 不计入）
    confidences: list = field(default_factory=list)   # 与 token_ids 一一对应的置信度
    hidden_vectors: list = field(default_factory=list)  # 与 token_ids 一一对应的隐藏向量
    prev_attn: torch.Tensor = None                    # 上一个 token 的多头平均注意力向量
    geomean_confidence: float = 1.0                   # 本步骤置信度几何平均（提前终止也留存，供规则9）
    cache_len_at_end: int = 0                         # 结束时刻的 KV 长度（裁掉占位 token 用）
    final_logits: torch.Tensor = None                 # 结束时刻的下一 token 分布（供下一步骤起点）

    @property
    def finished(self):
        return self.status != STATUS_RUNNING

    def compute_geomean(self, extra_confidence=None):
        confs = list(self.confidences)
        if extra_confidence is not None:
            confs.append(extra_confidence)
        if not confs:
            return 1.0  # 例如首 token 即 EOS：无已产出 token，按"确信地结束"处理
        logs = [math.log(max(c, config.GEO_MEAN_EPS)) for c in confs]
        return math.exp(sum(logs) / len(logs))

    def finalize(self, status, end_reason, cache_len, final_logits, extra_confidence=None):
        """结束该路径：记录标记、几何平均置信度、缓存长度与最终分布；
        extra_confidence —— 触发提前终止的那个 token 自身的置信度，计入快照
        （规则6：每个 token 生成时都取 top-1 概率；否则首 token 即死亡的路径
        会得到空序列约定的 1.0，在规则9 选取中不合理地压过生成了内容的路径）。
        若为提前终止，按规则6/7 删除置信度序列与隐藏状态序列
        （KV 缓存与几何平均值按既定方案在步骤内暂留，仅供规则9兜底选取）。"""
        self.status = status
        self.end_reason = end_reason
        self.cache_len_at_end = int(cache_len)
        self.final_logits = final_logits
        self.geomean_confidence = self.compute_geomean(extra_confidence)
        if status == STATUS_EARLY:
            self.confidences = []
            self.hidden_vectors = []
            self.prev_attn = None


# ---------------------------------------------------------------------------
# 步骤 / 生成 结果容器
# ---------------------------------------------------------------------------
@dataclass
class StepResult:
    cache: DynamicCache            # 胜者的 KV 缓存（已裁剪、已按需拼接 <self-check>），batch=1
    next_logits: torch.Tensor      # 进入下一步骤的起始分布 (1, vocab)
    output_token_ids: list         # 本步骤计入最终回复的 token
    end_generation: bool           # True = 整段生成到此结束（胜者以 EOS 结束且未拼接 self-check）
    info: dict                     # 本步骤的诊断信息


@dataclass
class GenerationResult:
    text: str
    token_ids: list
    stop_reason: str
    steps: list


# ---------------------------------------------------------------------------
# 引擎主体
# ---------------------------------------------------------------------------
class MPVDHSEngine:
    def __init__(self):
        print(f"[MPVDHS] 加载模型: {config.MODEL_PATH}")
        self.device = self._pick_device()
        self.dtype = self._pick_dtype(self.device)
        print(f"[MPVDHS] 推理设备: {self.device} | 权重精度: {self.dtype}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.MODEL_PATH, trust_remote_code=config.TRUST_REMOTE_CODE
        )
        load_kwargs = dict(
            attn_implementation=config.ATTN_IMPLEMENTATION,  # 必须 eager 才能拿注意力权重
            trust_remote_code=config.TRUST_REMOTE_CODE,
        )
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                config.MODEL_PATH, dtype=self.dtype, **load_kwargs
            )
        except TypeError:  # 旧版 transformers 只认 torch_dtype
            self.model = AutoModelForCausalLM.from_pretrained(
                config.MODEL_PATH, torch_dtype=self.dtype, **load_kwargs
            )
        self.model.to(self.device)
        self.model.eval()

        self.eos_ids = self._collect_eos_ids()
        self.self_check_ids = self._prepare_self_check()

        if config.SEED is not None:
            torch.manual_seed(config.SEED)

        print(
            f"[MPVDHS] 初始化完成: 路径数 n={config.NUM_PATHS}, "
            f"temperature={config.TEMPERATURE}, attn={config.ATTN_IMPLEMENTATION}"
        )

    # ------------------------------------------------------------------
    # 初始化辅助
    # ------------------------------------------------------------------
    @staticmethod
    def _pick_device():
        mode = str(config.DEVICE_MODE).lower()
        if mode == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("DEVICE_MODE='cuda' 但当前没有可用的 CUDA GPU")
            return torch.device("cuda")
        if mode == "cpu":
            return torch.device("cpu")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _pick_dtype(device):
        if device.type == "cuda":
            table = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
            key = str(config.GPU_DTYPE).lower()
        else:
            table = {"fp32": torch.float32, "bf16": torch.bfloat16}
            key = str(config.CPU_DTYPE).lower()
        if key not in table:
            raise ValueError(f"未知的精度配置: {key}")
        return table[key]

    def _collect_eos_ids(self):
        """汇总 tokenizer 与 generation_config 里的所有 EOS id（如 Qwen 的 <|im_end|>/<|endoftext|>）。"""
        ids = set()
        if self.tokenizer.eos_token_id is not None:
            ids.add(int(self.tokenizer.eos_token_id))
        gen_cfg = getattr(self.model, "generation_config", None)
        eos = getattr(gen_cfg, "eos_token_id", None)
        if eos is not None:
            if isinstance(eos, (list, tuple)):
                ids.update(int(x) for x in eos)
            else:
                ids.add(int(eos))
        return ids

    def _prepare_self_check(self):
        """把 <self-check> 准备成要强制拼接的 token id 序列（规则8/9）。"""
        if config.SELF_CHECK_AS_SPECIAL:
            added = self.tokenizer.add_special_tokens(
                {"additional_special_tokens": [config.SELF_CHECK_TOKEN]}
            )
            if added:
                self.model.resize_token_embeddings(len(self.tokenizer))
            token_id = self.tokenizer.convert_tokens_to_ids(config.SELF_CHECK_TOKEN)
            print(f"[MPVDHS] <self-check> 已注册为特殊 token id={token_id}（embedding 已扩展）")
            return [int(token_id)]
        ids = self.tokenizer.encode(config.SELF_CHECK_TOKEN, add_special_tokens=False)
        if not ids:
            raise ValueError(f"<self-check> 字符串无法分词: {config.SELF_CHECK_TOKEN!r}")
        print(f"[MPVDHS] <self-check> 按普通文本分词: {ids}")
        return [int(x) for x in ids]

    # ------------------------------------------------------------------
    # 对外主入口：传入对话历史（符合大语言模型格式的上下文），返回本轮回复
    # ------------------------------------------------------------------
    def generate(self, messages) -> GenerationResult:
        if not messages:
            raise ValueError("messages 不能为空")

        # 规则4：自动读取对话模板
        prompt_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )
        if torch.is_tensor(prompt_ids):
            prompt_ids = prompt_ids.reshape(-1).tolist()

        cache, start_logits = self._prefill(prompt_ids)  # 无上一步缓存 -> 从头预填充

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
            if result.end_generation:  # 规则：胜者以 EOS 结束 -> 整段生成完成
                stop_reason = "eos"
                break

        text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return GenerationResult(text=text, token_ids=output_ids, stop_reason=stop_reason, steps=steps_info)

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _prefill(self, prompt_ids):
        """整段提示词一次性前向，得到初始 KV 缓存与首个 token 的分布。"""
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        out = self.model(input_ids=input_ids, use_cache=True)
        return out.past_key_values, out.logits[:, -1, :].float()

    @torch.inference_mode()
    def _append_token_ids(self, cache, token_ids):
        """把若干 token（<self-check>）强制拼接到缓存上，不经过采样。
        返回 (新缓存, 拼接后的下一 token 分布)。"""
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        out = self.model(input_ids=input_ids, past_key_values=cache, use_cache=True)
        return out.past_key_values, out.logits[:, -1, :].float()

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _run_step(self, base_cache, start_logits, step_idx) -> StepResult:
        """规则4~9 的单步骤全流程。"""
        n = config.NUM_PATHS
        prefix_len = get_cache_seq_len(base_cache)
        paths = [PathState(i) for i in range(n)]
        batch_cache = replicate_cache(base_cache, n)

        # ---------- 首个 token：n 条路径从同一分布(start_logits)各自采样 ----------
        first_tokens, first_confs = sample_batch(start_logits.repeat(n, 1))
        # 已结束路径后续改喂占位 token（只为凑批，最后会从缓存中裁掉）
        dummy_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0
        cur_tokens = torch.full((n, 1), int(dummy_id), dtype=torch.long, device=self.device)
        for p in paths:
            token = int(first_tokens[p.index])
            conf = float(first_confs[p.index])
            if token in self.eos_ids:                       # 首 token 即 EOS
                p.finalize(STATUS_EOS, END_EOS, prefix_len, start_logits)
            elif conf < config.CONFIDENCE_THRESHOLD:        # 首 token 即低置信
                p.finalize(STATUS_EARLY, END_CONFIDENCE, prefix_len, start_logits, extra_confidence=conf)
            else:
                p.token_ids.append(token)
                p.confidences.append(conf)
                cur_tokens[p.index, 0] = token

        # ---------- 逐 token 解码：每轮给每条正在运行的路径喂一个 token ----------
        t = 1  # 本步骤已生成的 token 数
        while any(not p.finished for p in paths):
            out = self.model(
                input_ids=cur_tokens,
                past_key_values=batch_cache,
                use_cache=True,
                output_attentions=True,   # 规则5
                output_hidden_states=True,  # 规则7
            )
            batch_cache = out.past_key_values
            kv_len = get_cache_seq_len(batch_cache)          # = prefix_len + t
            logits = out.logits[:, -1, :].float()            # (n, vocab) 预测下一 token
            attn_last_layer = out.attentions[-1]             # (n, heads, 1, kv_len) 最后一层
            hidden_last_layer = out.hidden_states[-1]        # (n, 1, hidden) 最后一层

            next_tokens, next_confs = sample_batch(logits)

            for p in paths:
                if p.finished:
                    continue  # 已结束的路径本轮喂的是占位 token，输出全部忽略
                i = p.index

                # 规则7：把该 token 的最后一层隐藏状态拼进向量序列
                hidden_vec = hidden_last_layer[i, 0, :].float()
                p.hidden_vectors.append(hidden_vec)

                # 规则5：与上一个 token 的注意力向量比较"公共历史部分"
                #   上一 token 的向量覆盖 0..pos(t-1)，本 token 覆盖 0..pos(t)，
                #   双方都去掉自己所在位置后，公共部分 = 0..pos(t-1)-1，
                #   即 prev[:-1] 与 cur[:len(prev)-1]。
                attn_vec = attn_last_layer[i, :, 0, :].mean(dim=0).float()  # 多头取平均
                if p.prev_attn is not None:
                    common_len = p.prev_attn.shape[0] - 1
                    if common_len >= config.ATTN_MIN_COMMON_LEN:
                        sim = cosine_similarity(p.prev_attn[:common_len], attn_vec[:common_len])
                        if sim < config.ATTN_SIM_THRESHOLD:
                            p.finalize(STATUS_STEP_DONE, END_ATTENTION, kv_len, logits[i : i + 1])
                            continue
                p.prev_attn = attn_vec  # 滑动窗口：只保留最近一个（满足"用过两次即删"）

                # 步内 token 上限：强制按"步骤完成"结束（不再采样下一个 token）
                if t >= config.MAX_STEP_TOKENS:
                    p.finalize(STATUS_STEP_DONE, END_MAX_TOKENS, kv_len, logits[i : i + 1])
                    continue

                # 规则6：采样下一 token；EOS 优先于置信度检查
                token = int(next_tokens[i])
                conf = float(next_confs[i])
                if token in self.eos_ids:
                    p.finalize(STATUS_EOS, END_EOS, kv_len, logits[i : i + 1])  # EOS 不计入文本/缓存
                elif conf < config.CONFIDENCE_THRESHOLD:
                    p.finalize(STATUS_EARLY, END_CONFIDENCE, kv_len, logits[i : i + 1], extra_confidence=conf)
                else:
                    p.token_ids.append(token)
                    p.confidences.append(conf)
                    cur_tokens[i, 0] = token
            t += 1

        return self._resolve_step(paths, batch_cache, step_idx)

    # ------------------------------------------------------------------
    def _resolve_step(self, paths, batch_cache, step_idx) -> StepResult:
        """步骤末尾裁决（规则8/9）：选胜者、判分歧、清场。"""
        # 未提前终止的路径按几何平均置信度排序（规则8）
        survivors = [p for p in paths if p.status in (STATUS_STEP_DONE, STATUS_EOS)]
        survivors.sort(key=lambda p: p.geomean_confidence, reverse=True)

        verdict, self_check = "", False
        head_sim = tail_sim = None

        if len(survivors) >= 2:
            top1, top2 = survivors[0], survivors[1]
            if not top1.hidden_vectors or not top2.hidden_vectors:
                head_sim = tail_sim = 1.0
                verdict = "无分歧（序列为空，跳过DTW）"
            else:
                head_sim, tail_sim = compare_hidden_sequences(top1.hidden_vectors, top2.hidden_vectors)
                if head_sim < config.HEAD_SIM_THRESHOLD:
                    verdict = "创造性分歧"
                elif tail_sim < config.TAIL_SIM_THRESHOLD:
                    verdict = "错误性分歧"
                    self_check = True
                else:
                    verdict = "无分歧"
            kept = top1
        elif len(survivors) == 1:
            # 用户决定：仅剩 1 条路径时按规则9处理（保留它 + 强制 <self-check>）
            kept = survivors[0]
            verdict = "仅剩1条路径（按规则9处理）"
            self_check = True
        else:
            # 规则9：全部提前终止 -> 用暂存的几何平均置信度选出最高者
            kept = max(paths, key=lambda p: p.geomean_confidence)
            verdict = "全部路径提前终止（规则9）"
            self_check = True

        output_ids = list(kept.token_ids)  # 胜者本步骤的 token 计入最终回复

        # 取出胜者的 KV 缓存（裁掉结束后喂入的占位 token），清除其他路径缓存
        kept_cache = slice_path_cache(batch_cache, kept.index, kept.cache_len_at_end)

        if self_check:
            kept_cache, next_logits = self._append_token_ids(kept_cache, self.self_check_ids)
            if config.SELF_CHECK_SHOW_IN_OUTPUT:
                output_ids = output_ids + list(self.self_check_ids)
        else:
            next_logits = kept.final_logits

        end_generation = (kept.status == STATUS_EOS) and not self_check

        info = {
            "step": step_idx,
            "paths": [
                {
                    "index": p.index,
                    "status": p.status,
                    "end_reason": p.end_reason,
                    "num_tokens": len(p.token_ids),
                    "geomean_confidence": round(p.geomean_confidence, 4),
                }
                for p in paths
            ],
            "verdict": verdict,
            "head_sim": None if head_sim is None else round(head_sim, 4),
            "tail_sim": None if tail_sim is None else round(tail_sim, 4),
            "kept_path": kept.index,
            "self_check": self_check,
            "step_output_tokens": len(output_ids),
        }
        if config.VERBOSE:
            self._print_step_info(info)

        return StepResult(
            cache=kept_cache,
            next_logits=next_logits,
            output_token_ids=output_ids,
            end_generation=end_generation,
            info=info,
        )

    @staticmethod
    def _print_step_info(info):
        print(f"[步骤 {info['step']}]")
        for p in info["paths"]:
            print(
                f"  路径{p['index']}: {p['status']}({p['end_reason'] or '-'}) "
                f"tokens={p['num_tokens']} geoConf={p['geomean_confidence']}"
            )
        sim = ""
        if info["head_sim"] is not None:
            sim = f" | head_sim={info['head_sim']} tail_sim={info['tail_sim']}"
        suffix = " + <self-check>" if info["self_check"] else ""
        print(
            f"  裁决: {info['verdict']}{sim} -> 保留路径{info['kept_path']}{suffix} "
            f"| 本步输出 {info['step_output_tokens']} token"
        )
