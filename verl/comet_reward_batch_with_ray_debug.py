# -*- coding: utf-8 -*-
"""
comet_reward_batch_with_ray_debug.py

用途：
- 既支持本地加载 COMET / XCOMET 评分（与你现有逻辑一致）
- 也支持在绑定了 VERL 的远端 Reward Worker (MetricRewardWorker) 时，直接把 pair/triplet 发给远端打分
- 远端异常/未绑定时会自动回退本地计算
- 对 BatchRewardManager 和 Naive/DAPO 单条调用均保持原有签名与返回格式
"""

import os
import re
import logging
from typing import List, Dict, Any, Optional
import torch
from tqdm import tqdm
from comet import load_from_checkpoint
import numpy as np

# ----------------- 远端 Worker 绑定（可选） -----------------
_RM_WG = None
def bind_rm_wg(wg):
    """由 main_ppo 在 trainer.init_workers() 之后调用。"""
    global _RM_WG
    _RM_WG = wg
    print("[Reward][bind] rm_wg bound:", type(wg).__name__)

def _use_remote_worker() -> bool:
    ok = _RM_WG is not None
    if not ok:
        print("[Reward] remote worker NOT bound. Using LOCAL fallback (CPU if driver has no GPU).")
    return ok

# ---------- 降噪 ----------
for name in logging.root.manager.loggerDict:
    try:
        logging.getLogger(name).setLevel(logging.WARNING)
    except Exception:
        pass

# ================== 配置（环境变量可覆盖） ==================
_WORD_LEVEL_BATCH = int(os.getenv("WORD_LEVEL_BATCH", "32"))  # xcomet batch size
WORD_QE_MODE      = os.getenv("WORD_QE_MODE", "only").lower() # off | only | add
WORD_QE_WEIGHT    = float(os.getenv("WORD_QE_WEIGHT", "0.2")) # add 模式下加权
_WORD_QE_CKPT     = os.getenv(
    "WORD_QE_CKPT",
    "/mnt/data1/users/4xin/hf/hub/models--Unbabel--XCOMET-XL/snapshots/6a123c5e8e6dccab25e5fcffa3c8b417abadb462/checkpoints/model.ckpt",
)

_COMET_BATCH = int(os.getenv("COMET_BATCH", "32"))
_COMET_CKPT  = os.getenv(
    "COMET_CKPT",
    "/mnt/data1/users/4xin/hf/hub/models--Unbabel--wmt23-cometkiwi-da-xl/"
    "snapshots/33858b2239a139d497d9c74952c88b89a8c06213/checkpoints/model.ckpt",
)

# ================== 本地模型（惰性加载） ==================
_word_qe_model, _word_qe_device = None, None
def _load_word_qe_model():
    """懒加载 XCOMET（legacy word-level / triplet 评分也可跑）。"""
    global _word_qe_model, _word_qe_device
    if WORD_QE_MODE == "off":
        return None, None
    if _word_qe_model is not None:
        return _word_qe_model, _word_qe_device
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        m = load_from_checkpoint(_WORD_QE_CKPT).to(dev)
        m.eval()
        _word_qe_model, _word_qe_device = m, dev
        print(f"[WORD-QE] loaded on {dev}")
    except Exception as e:
        print(f"[WORD-QE] load failed: {e}")
        _word_qe_model, _word_qe_device = None, None
    return _word_qe_model, _word_qe_device

_comet_model, _comet_device = None, None
def _load_comet_model():
    """懒加载 COMET（pair 评分）。"""
    global _comet_model, _comet_device
    if _comet_model is not None:
        return _comet_model, _comet_device
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        m = load_from_checkpoint(_COMET_CKPT).to(dev)
        m.eval()
        _comet_model, _comet_device = m, dev
        print(f"[COMET] loaded on {dev}")
    except Exception as e:
        print(f"[COMET] load failed: {e}")
        _comet_model, _comet_device = None, None
    return _comet_model, _comet_device

# ================== 工具函数 ==================
def compute_bleu(lg_pair: str, ref: str, pred: str) -> float:
    import sacrebleu
    pred = pred if isinstance(pred, str) else ""
    tgt_lang = lg_pair.split("-")[1]
    tokenize = "zh" if tgt_lang == "zh" else "ja-mecab" if tgt_lang == "ja" else "13a"
    bleu = sacrebleu.sentence_bleu(pred, [ref], lowercase=True, tokenize=tokenize)
    return float(bleu.score)

def extract_solution(solution_str: str) -> Optional[str]:
    pat = r"<translate>(.*?)</translate>"
    m = list(re.finditer(pat, solution_str, re.DOTALL))
    if not m:
        return None
    return m[-1].group(1).strip()

def validate_response_structure(s: str) -> bool:
    tags = {
        "think_start": ("<think>", 1),
        "think_end": ("</think>", 1),
        "ans_start": ("<translate>", 1),
        "ans_end": ("</translate>", 1),
    }
    ok, pos = True, {}
    for k, (t, exp) in tags.items():
        c = s.count(t)
        pos[k] = s.find(t)
        if c != exp:
            ok = False
    if (pos["think_start"] > pos["think_end"]
        or pos["think_end"] > pos["ans_start"]
        or pos["ans_start"] > pos["ans_end"]):
        ok = False
    return ok

def _merge_remote_metric_out(out):
    """
    RayWorkerGroup.score(...) 可能返回：
      - dict: {"comet":[...], "xcomet":[...]}
      - list[dict]: 每个 DP rank 一份，需要把同名 metric 的列表拼接
    """
    if isinstance(out, dict):
        return out
    if isinstance(out, list):
        merged = {}
        for item in out:
            if not isinstance(item, dict):
                continue
            for k, v in item.items():
                merged.setdefault(k, []).extend(list(v))
        return merged
    return {}


# ================== 本地打分（作为远端失败/未绑定的回退） ==================
def score_word_level(items: List[Dict[str, str]]) -> List[float]:
    """XCOMET / Word-level 接口；items 为 triplet: {"src","mt","ref"}。"""
    model, dev = _load_word_qe_model()
    if model is None:
        return [0.0] * len(items)
    use_gpu = 1 if (dev and dev.startswith("cuda") and torch.cuda.is_available()) else 0
    out = model.predict(items, batch_size=_WORD_LEVEL_BATCH, gpus=use_gpu)
    return out.get("scores") if isinstance(out, dict) else list(out.scores)

def score_comet(items: List[Dict[str, str]]) -> List[float]:
    """COMET；items 为 pair: {"src","mt"}。"""
    model, dev = _load_comet_model()
    if model is None:
        return [0.0] * len(items)
    use_gpu = 1 if (dev and dev.startswith("cuda") and torch.cuda.is_available()) else 0
    out = model.predict(items, batch_size=_COMET_BATCH, gpus=use_gpu)
    return out.get("scores") if isinstance(out, dict) else list(out.scores)

# ================== 单条评分 ==================
def compute_score_single(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None,
    compute_val_reward: bool = False,
) -> float:
    lg_pair = extra_info.get("lg", "en-zh") if extra_info else "en-zh"
    src_text = extra_info.get("source", ground_truth) if extra_info else ground_truth

    ok = validate_response_structure(solution_str)
    ans = extract_solution(solution_str)

    if not ok or ans is None:
        if compute_val_reward:
            out = {"score": -3.0, "format_score": -3.0, "bleu_score": float("nan"), "comet_score": float("nan")}
            if WORD_QE_MODE in ["only", "add"]:
                out["word_level_qe"] = float("nan")
            return out
        return -3.0

    fmt = 1.0
    
    comet_val, xqe_val = 0.0, 0.0
    comet_data  = [{"src": src_text, "mt": ans}] if (WORD_QE_MODE in ["off", "add"] or compute_val_reward) else None
    xcomet_data = [{"src": src_text, "mt": ans, "ref": ground_truth}] if (WORD_QE_MODE in ["only", "add"]) else None


    if _use_remote_worker():
        need_metrics = []
        if comet_data is not None:
            need_metrics.append("comet")
        if xcomet_data is not None:
            need_metrics.append("xcomet")
        try:
            remote_out = _RM_WG.score(
                src_mt_pairs=comet_data,
                triplets=xcomet_data,
                metrics=need_metrics
            )
            out = _merge_remote_metric_out(remote_out)
            if comet_data:
                lst = out.get("comet", [0.0])
                comet_val = float(lst[0])
            if xcomet_data:
                lst = out.get("xcomet", [0.0])
                xqe_val = float(lst[0])
        except Exception as e:
            print(f"[Reward][single] remote failed, fallback local: {type(e).__name__}: {e}")
            if comet_data:
                comet_val = score_comet(comet_data)[0]
            if xcomet_data:
                xqe_val = score_word_level(xcomet_data)[0]
    else:
        raise RuntimeError("[Reward] Remote RM worker not bound. Aborting to avoid CPU fallback.")
        # if comet_data:
        #     comet_val = score_comet(comet_data)[0]
        # if xcomet_data:
        #     xqe_val = score_word_level(xcomet_data)[0]

    if compute_val_reward:
        bleu_val = compute_bleu(lg_pair, ground_truth, ans) / 100.0
        res = {
            "score": float("nan"),
            "format_score": fmt,
            "bleu_score": bleu_val,
            "comet_score": comet_val,
        }
        if WORD_QE_MODE in ["only", "add"]:
            res["word_level_qe"] = xqe_val
        total = fmt + bleu_val + comet_val
        if WORD_QE_MODE == "add":
            total += WORD_QE_WEIGHT * xqe_val
        res["score"] = total
        return res

    if WORD_QE_MODE == "only":
        return fmt + xqe_val
    elif WORD_QE_MODE == "add":
        bleu_val = compute_bleu(lg_pair, ground_truth, ans) / 100.0
        return fmt + bleu_val + comet_val + WORD_QE_WEIGHT * xqe_val
    else:  # off
        bleu_val = compute_bleu(lg_pair, ground_truth, ans) / 100.0
        return fmt + bleu_val + comet_val

# ================== 批量评分（BatchRewardManager 用） ==================
def compute_score_batch(
    data_sources: List[str],
    solution_strs: List[str],
    ground_truths: List[str],
    extra_infos: Optional[List[Optional[Dict[str, Any]]]] = None,
    compute_val_reward: bool = False,
    micro_batch_size: int = 8,  # 保留签名
) -> List[float]:
    if extra_infos is None:
        extra_infos = [None] * len(solution_strs)

    data_list: List[Dict[str, Any]] = []
    final_scores: List[float] = []
    invalid_items: List[int] = []

    print(f"Processing batch of {len(solution_strs)} items...")
    print("lens:", len(data_sources), len(solution_strs), len(ground_truths), len(extra_infos))

    fmt = 1.0  # format score 固定为 1.0
    for i in tqdm(range(len(solution_strs)), desc="checking format + building"):
        sol = solution_strs[i]
        gt  = ground_truths[i]
        info = extra_infos[i]
        lg_pair = info.get("lg", "en-zh") if info else "en-zh"
        src_text = info.get("source", gt) if info else gt
        ans = extract_solution(sol)

        if not validate_response_structure(sol) or ans is None:
            invalid_items.append(i)
            if compute_val_reward:
                out = {"score": -3.0, "format_score": -3.0, "bleu_score": float("nan"), "comet_score": float("nan")}
                if WORD_QE_MODE in ["only", "add"]:
                    out["word_level_qe"] = float("nan")
                final_scores.append(out)
            else:
                final_scores.append(-3.0)
            continue

        if compute_val_reward:
            out = {"score": float("nan"), "format_score": fmt, "bleu_score": float("nan"), "comet_score": float("nan")}
            if WORD_QE_MODE in ["only", "add"]:
                out["word_level_qe"] = float("nan")
            final_scores.append(out)
        else:
            final_scores.append(float("nan"))

        if WORD_QE_MODE in ["off", "add"] or compute_val_reward:
            bleu = compute_bleu(lg_pair, gt, ans)
            entry = {
                "src_mt_pair": {"src": src_text, "mt": ans},
                "format_score": fmt,
                "bleu_score": bleu,
                "index": i,
            }
            if WORD_QE_MODE in ["add", "only"]:
                entry["triplet"] = {"src": src_text, "mt": ans, "ref": gt}
            data_list.append(entry)
        else:  # trian & only
            data_list.append({
                "triplet": {"src": src_text, "mt": ans, "ref": gt},
                "format_score": fmt,
                "src_mt_pair": {"src": src_text, "mt": ans},
                "index": i,
            })

    print(f"invalid items: {len(invalid_items)} / {len(solution_strs)}")

    if len(data_list) > 0:
        comet_data, xcomet_data = None, None
        if WORD_QE_MODE in ["off", "add"] or compute_val_reward:
            comet_data = [x["src_mt_pair"] for x in data_list]

        if WORD_QE_MODE in ["only", "add"]:
            xcomet_data = [x["triplet"] for x in data_list]

        if _use_remote_worker():
            need_metrics = []
            if comet_data is not None:
                need_metrics.append("comet")
            if xcomet_data is not None:
                need_metrics.append("xcomet")

            try:
                remote_out = _RM_WG.score(
                src_mt_pairs=comet_data,
                triplets=xcomet_data,
                metrics=need_metrics
                )
                out = _merge_remote_metric_out(remote_out)

                if WORD_QE_MODE in ["off", "add"] or compute_val_reward:
                    comet_scores = out["comet"]
                if WORD_QE_MODE in ["only", "add"]:
                    word_qe_scores = out["xcomet"]

            except Exception as e:
                print(f"[Reward][batch] remote failed, fallback local: {type(e).__name__}: {e}")
                if WORD_QE_MODE in ["off", "add"] or compute_val_reward:
                    comet_scores = score_comet(comet_data)
                if WORD_QE_MODE in ["only", "add"]:
                    word_qe_scores = score_word_level(xcomet_data)
        else:
            raise RuntimeError("[Reward] Remote RM worker not bound. Aborting to avoid CPU fallback.")
            # if WORD_QE_MODE in ["off", "add"] or compute_val_reward:
            #     comet_scores = score_comet(comet_data)
            # if WORD_QE_MODE in ["only", "add"]:
            #     word_qe_scores = score_word_level(word_qe_data)

        for i, item in enumerate(data_list):
            j = item["index"]
            if WORD_QE_MODE in ["only", "add"]:
                if compute_val_reward:
                    final_scores[j]["format_score"] = fmt
                    final_scores[j]["bleu_score"]   = item["bleu_score"] / 100.0 if "bleu_score" in item else float("nan")
                    final_scores[j]["comet_score"]  = comet_scores[i] if comet_scores else float("nan")
                    final_scores[j]["word_level_qe"]= word_qe_scores[i] if word_qe_scores else float("nan")
                    final_scores[j]["score"]        = final_scores[j]["format_score"] + final_scores[j]["bleu_score"] + final_scores[j]["comet_score"]
                    if WORD_QE_MODE == "add":
                        final_scores[j]["score"] += WORD_QE_WEIGHT * final_scores[j]["word_level_qe"]
                else:
                    if WORD_QE_MODE == "only":
                        final_scores[j] = fmt + word_qe_scores[i]
                    else:  # add
                        final_scores[j] = fmt + item["bleu_score"] / 100.0 + comet_scores[i] + WORD_QE_WEIGHT * word_qe_scores[i]
            else:  # off
                if compute_val_reward:
                    final_scores[j]["format_score"] = fmt
                    final_scores[j]["bleu_score"]   = item["bleu_score"] / 100.0
                    final_scores[j]["comet_score"]  = comet_scores[i]
                    final_scores[j]["score"]        = final_scores[j]["format_score"] + final_scores[j]["bleu_score"] + final_scores[j]["comet_score"]
                else:
                    final_scores[j] = fmt + item["bleu_score"] / 100.0 + comet_scores[i]

    print(f"Batch done: {len(final_scores)} scores")
    return final_scores

# ================== 统一入口（保持原签名） ==================
def compute_score(*args, **kwargs):
    """
    兼容三种调用：
    1) 批量（BatchRewardManager）：compute_score(data_sources=[], solution_strs=[], ground_truths=[], extra_infos=None, ...)
    2) 单条（位置）：compute_score(data_source, solution_str, ground_truth, extra_info=None)
    3) 单条（关键字）：compute_score(data_source=..., solution_str=..., ground_truth=..., extra_info=None)
    """

    if 'data_sources' in kwargs or 'solution_strs' in kwargs or 'ground_truths' in kwargs:
        results = compute_score_batch(
            kwargs.get('data_sources', []),
            kwargs.get('solution_strs', []),
            kwargs.get('ground_truths', []),
            kwargs.get('extra_infos', None),
            kwargs.get('compute_val_reward', False),
            kwargs.get('micro_batch_size', 8),
        )
    elif len(args) >= 3:
        results = compute_score_single(
            args[0], args[1], args[2],
            args[3] if len(args) > 3 else kwargs.get('extra_info', None),
            args[4] if len(args) > 4 else kwargs.get('compute_val_reward', False),
        )
    elif {'data_source', 'solution_str', 'ground_truth'} <= set(kwargs.keys()):
        results = compute_score_single(
            kwargs['data_source'],
            kwargs['solution_str'],
            kwargs['ground_truth'],
            kwargs.get('extra_info', None),
            kwargs.get('compute_val_reward', False),
        )
    else:
        raise ValueError(f"Invalid arguments for compute_score: args={args}, kwargs={kwargs}")
    
    return results

