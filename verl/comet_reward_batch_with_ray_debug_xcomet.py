# TODO: replace current wod level qe with xcomet sequence metrics.
# comet 无法识别ref，字典中不能包括reference


import re, os
import logging
from typing import List, Dict, Any, Optional
from comet import download_model, load_from_checkpoint
import torch
from tqdm import tqdm
import itertools, ray

# ---------- Logging: 降低外部包日志噪音 ----------
for name in logging.root.manager.loggerDict:
    try:
        logging.getLogger(name).setLevel(logging.WARNING)
    except Exception:
        pass

REWARD_MUST_USE_GPU = os.getenv("REWARD_MUST_USE_GPU", "true").lower() in {"1","true","yes"}
if REWARD_MUST_USE_GPU and not torch.cuda.is_available():
    raise RuntimeError(
        "[Reward] GPU is required but not available. "
        "Please set trainer.taskrunner_num_gpus=1 in config."
    )

_REWARD_ACTOR_HANDLES = []
_HANDLE_ROUND_ROBIN = None

## =====导入 Word-level QE ===== ##
# Word-level 开关与权重
_WORD_LEVEL_BATCH = int(os.getenv("WORD_LEVEL_BATCH", "32"))  # xcomet 批量预测的 batch size
WORD_QE_MODE = os.getenv("WORD_QE_MODE", "only").lower()   # off | only | add
WORD_QE_WEIGHT = float(os.getenv("WORD_QE_WEIGHT", "0.2"))  # add 模式下的加权
# word level 路径
_WORD_QE_CKPT = os.getenv(
    "WORD_QE_CKPT",
    "/mnt/data1/users/4xin/hf/hub/models--Unbabel--XCOMET-XXL/snapshots/873bac1b1c461e410c4a6e379f6790d3d1c7c214/checkpoints/model.ckpt",
)


# 惰性加载xcomet 模型
_word_qe_model = None
_word_qe_device = None

def _load_word_qe_model():
    """懒加载 legacy word-level 模型；根据可用性放到合适的 device。"""
    global _word_qe_model, _word_qe_device
    if WORD_QE_MODE == "off":
        return None, None
    if _word_qe_model is not None:
        return _word_qe_model, _word_qe_device

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = load_from_checkpoint(_WORD_QE_CKPT).to(dev)
        model.eval()
        _word_qe_model, _word_qe_device = model, dev
        print(f"[WORD-QE] xcomet model loaded on {dev}")
    except Exception as e:
        print(f"[WORD-QE] load failed: {e}")
        _word_qe_model, _word_qe_device = None, None
    return _word_qe_model, _word_qe_device


# ==================================================================
#   懒加载 COMET 序列级模型；默认 device 由 COMET_DEVICE 控制（cpu|cuda）
# ==================================================================
# ---------- 配置项（可用环境变量覆盖） ----------
_COMET_BATCH = int(os.getenv("COMET_BATCH", "32"))  # comet 批量预测的 batch size
_COMET_CKPT = os.getenv(
    "COMET_CKPT",
    "/mnt/data1/users/4xin/hf/hub/"
    "models--Unbabel--wmt23-cometkiwi-da-xl/"
    "snapshots/33858b2239a139d497d9c74952c88b89a8c06213/"
    "checkpoints/model.ckpt",
)

# 全局变量缓存模型  
_comet_model = None
_comet_device = None

def _load_comet_model():  
    """懒加载 comet 模型；根据可用性放到合适的 device。"""
    global _comet_model, _comet_device
    if _comet_model is not None:
        return _comet_model, _comet_device
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = load_from_checkpoint(_COMET_CKPT).to(dev)
        model.eval()
        _comet_model, _comet_device = model, dev
        print(f"[COMET] comet model loaded on {dev}")
    except Exception as e:
        print(f"[COMET] load failed: {e}")
        _comet_model, _comet_device = None, None
    return _comet_model, _comet_device

# =========================================================
#                     工具函数
# =========================================================
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
        # print("[Error] No valid <translate> tags found")
        return None
    return m[-1].group(1).strip()

def validate_response_structure(s: str) -> bool:
    # 必须有且仅有一次这四个标签，且顺序正确
    tags = {
        "think_start": ("<think>", 1),
        "think_end": ("</think>", 1),
        "ans_start": ("<translate>", 1),
        "ans_end": ("</translate>", 1),
    }
    ok = True
    pos = {}
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

def _avg_token_reward_from_tags(tags):
    """OK -> +1, BAD -> 0，其它（或缺失）按 OK 处理；返回句子平均分[-1,1]。"""
    if not isinstance(tags, list) or len(tags) == 0:
        return 0.0
    s = 0
    n = 0
    for t in tags:
        lab = str(t).upper() if t is not None else "OK"
        s += 0.0 if lab == "BAD" else 1.0
        n += 1
    return s / max(1, n)

def score_word_level(data: List[Dict[str, str]]) -> List[float]:
    """
    return model_output.scores: float in [0,1]
    - 若已注入 RewardActor，则优先用远程 GPU 推理（handle.score_xcomet.remote）。
    - 否则回退到本地 lazy-load 的 _word_qe_model.predict。
    """
    model, dev = _load_word_qe_model()
    if model is None:
        return [0.0] * len(data)

    use_gpu_flag = 1 if (dev and dev.startswith("cuda") and torch.cuda.is_available()) else 0
    try:
        out = model.predict(data, batch_size=_WORD_LEVEL_BATCH, gpus=use_gpu_flag)
        return out.get("scores")
    except Exception as e:
        raise RuntimeError(
            f"XCOMET model prediction failed. "
            f"batch_size={_WORD_LEVEL_BATCH}, gpus={use_gpu_flag}, "
            f"input_length={len(data)}.\n"
            f"Error: {type(e).__name__}: {e}"
        ) from e

def score_comet(data: List[Dict[str, str]]) -> List[float]:
    """
    return comet score: float
    """
    model, dev = _load_comet_model()
    if model is None:
        return [0.0] * len(data)

    use_gpu_flag = 1 if (dev and dev.startswith("cuda") and torch.cuda.is_available()) else 0
    try:
        out = model.predict(data, batch_size=_COMET_BATCH, gpus=use_gpu_flag)
        return out.get("scores")
    except Exception as e:
        raise RuntimeError(
            f"COMET model prediction failed. "
            f"batch_size={_COMET_BATCH}, gpus={use_gpu_flag}, "
            f"input_length={len(data)}.\n"
            f"Error: {type(e).__name__}: {e}"
        ) from e
        

def set_reward_clients(handles):
    """
    在 main_ppo.py 里 trainer.init_workers() 之后调用：
        reward_group = trainer.worker_groups[Role.RewardModel]
        set_reward_clients(reward_group.workers)
    传入的是 List[ray.ActorHandle]（RewardActor）
    """
    global _REWARD_ACTOR_HANDLES, _HANDLE_ROUND_ROBIN
    _REWARD_ACTOR_HANDLES = list(handles) if handles else []
    _HANDLE_ROUND_ROBIN = itertools.cycle(_REWARD_ACTOR_HANDLES) if _REWARD_ACTOR_HANDLES else None


def _has_remote_clients() -> bool:
    return ray is not None and ray.is_initialized() and bool(_REWARD_ACTOR_HANDLES)

def _pick_handle():
    assert _HANDLE_ROUND_ROBIN is not None, "RewardActor handles not set; call set_reward_clients(...) after init_workers()"
    return next(_HANDLE_ROUND_ROBIN)

# =========================================================
#                   单条评分（Naive / DAPO 单样本）
# =========================================================
def compute_score_single(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None,
    compute_val_reward: bool = False,
) -> float:
    lg_pair = extra_info.get("lg", "en-zh") if extra_info else "en-zh"
    src_text = extra_info.get("source", ground_truth) if extra_info else ground_truth

    format_score = validate_response_structure(solution_str)
    ans = extract_solution(solution_str)

    if not format_score:
        print("invalid format")
        if compute_val_reward:
            final_score = {
                "score": -3.0,
                "format_score": -3.0,
                "bleu_score": float("nan"),
                "comet_score": float("nan")
            }
            if WORD_QE_MODE in ["only", "add"]:
                final_score['word_level_qe'] = float("nan")
        else:
            final_score = -3.0
        return final_score
    
    if ans is None:
        print("format score is 1.0 but no <translate> tag found in completion")
        if compute_val_reward:
            final_score = {
                "score": -3.0,
                "format_score": -3.0,
                "bleu_score": float("nan"),
                "comet_score": float("nan")
            }
            if WORD_QE_MODE in ["only", "add"]:
                final_score['word_level_qe'] = float("nan")
        else:
            final_score = -3.0
        return final_score

    # 格式正确时：
    fmt = 1.0
    # construct data used for xcomet
    word_qe_data = None
    if WORD_QE_MODE in ["only", "add"]:
        word_qe_data = [{"src": src_text, "mt": ans, "ref": ground_truth}] # 因为要计算bleu score，所以ground truth肯定存在

    # construct data used for comet
    comet_data = None
    if WORD_QE_MODE in ["off", "add"] or compute_val_reward:
        comet_data = [{"src": src_text, "mt": ans}]

    # 计算指标值
    if WORD_QE_MODE in ["only", "add"]:
        if compute_val_reward:
            final_score = {
                "score": float("nan"),
                "format_score": fmt,
                "bleu_score": compute_bleu(lg_pair, ground_truth, ans) / 100.0,
                "comet_score": score_comet(comet_data)[0],
                "word_level_qe": score_word_level(word_qe_data)[0]
            }
            final_score['score'] = final_score['format_score'] + final_score['bleu_score'] + final_score['comet_score']
            if WORD_QE_MODE == "add":
                final_score['score'] += WORD_QE_WEIGHT * final_score['word_level_qe']
        else:
            if WORD_QE_MODE == "only":
                final_score = fmt + score_word_level(word_qe_data)[0]
            else: # WORD_QE_MODE == "add":
                final_score = fmt + compute_bleu(lg_pair, ground_truth, ans) / 100.0 + score_comet(comet_data)[0] + WORD_QE_WEIGHT * score_word_level(word_qe_data)[0]
    else: # WORD_QE_MODE == "off"
        if compute_val_reward:
            final_score = {
                "score": float("nan"),
                "format_score": fmt,
                "bleu_score": compute_bleu(lg_pair, ground_truth, ans) / 100.0,
                "comet_score": score_comet(comet_data)[0]
            }
            final_score['score'] = final_score['format_score'] + final_score['bleu_score'] + final_score['comet_score']
        else:
            final_score =  fmt + compute_bleu(lg_pair, ground_truth, ans) / 100.0 + score_comet(comet_data)[0]

    return final_score

# =========================================================
#                   批量评分（BatchRewardManager）
# =========================================================
def compute_score_batch(
    data_sources: List[str],
    solution_strs: List[str],
    ground_truths: List[str],
    extra_infos: Optional[List[Optional[Dict[str, Any]]]] = None,
    compute_val_reward: bool = False, 
    micro_batch_size: int = 8,  # 未使用（一次性送 actor，actor 内部再 batch）
) -> List[float]:
    if extra_infos is None:
        extra_infos = [None] * len(solution_strs)

    triplet_list = [] # 用于存放计算指标的数据
    final_scores: List[float] = []
    invalid_items: List[int] = []

    print(f"Processing batch of {len(solution_strs)} items...")
    print("data_sources", len(data_sources),
          "solution_strs", len(solution_strs),
          "ground_truths", len(ground_truths),
          "extra_infos", len(extra_infos))

    fmt = 1.0
    for i in tqdm(range(len(solution_strs)), desc="checking format and building triplets"):
        sol = solution_strs[i]
        gt = ground_truths[i]
        info = extra_infos[i]
        lg_pair = info.get("lg", "en-zh") if info else "en-zh"
        src_text = info.get("source", gt) if info else gt
        ans = extract_solution(sol)

        if not validate_response_structure(sol):
            invalid_items.append(i)
            if compute_val_reward:
                final_score = {
                    "score": -3.0,
                    "format_score": -3.0,
                    "bleu_score": float("nan"),
                    "comet_score": float("nan")
                }
                if WORD_QE_MODE in ["only", "add"]:
                    final_score['word_level_qe'] = float("nan")
                final_scores.append(final_score)
            else:
                final_scores.append(-3.0)
            continue
        
        if ans is None:
            invalid_items.append(i)
            if compute_val_reward:
                final_score = {
                    "score": -3.0,
                    "format_score": -3.0,
                    "bleu_score": float("nan"),
                    "comet_score": float("nan")
                }
                if WORD_QE_MODE in ["only", "add"]:
                    final_score['word_level_qe'] = float("nan")
                final_scores.append(final_score)
            else:
                final_scores.append(-3.0)
            continue

        # 当生成答案格式正确时：
        if compute_val_reward:
            final_score = {
                "score": fmt,
                "format_score": fmt,
                "bleu_score": float("nan"),
                "comet_score": float("nan")
            }
            if WORD_QE_MODE in ["only", "add"]:
                final_score['word_level_qe'] = float("nan")
            final_scores.append(final_score)
        else:
            final_scores.append(fmt)

        if WORD_QE_MODE in ["off", "add"] or compute_val_reward:
            bleu = compute_bleu(lg_pair, gt, ans)
            if WORD_QE_MODE == "off":
                triplet_list.append({
                    "src_mt_pair": {"src": src_text, "mt": ans},
                    "format_score": fmt,
                    "bleu_score": bleu,
                    "index": i,
                })
            else: # WORD_QE_MODE == "add":
                triplet_list.append({
                    "triplet": {"src": src_text, "mt": ans, "ref": gt},
                    "src_mt_pair": {"src": src_text, "mt": ans},
                    "format_score": fmt,
                    "bleu_score": bleu,
                    "index": i,
                })
        else: # WORD_QE_MODE == "only":
            triplet_list.append({
                "triplet": {"src": src_text, "mt": ans, "ref": gt},
                "src_mt_pair": {"src": src_text, "mt": ans},
                "index": i,
            })
    print(f"invalid items number {len(invalid_items)} / {len(solution_strs)}")
    
    if triplet_list: # 存在生成数据format合格
        comet_data= [x["src_mt_pair"] for x in triplet_list]
        word_qe_data = [x["triplet"] for x in triplet_list]
        fmt = 1.0  # 上面已验证结构
        
        # 计算comet score
        comet_scores = []
        if WORD_QE_MODE in ["off", "add"] or compute_val_reward:
            scores = score_comet(comet_data)
            comet_scores.extend(scores)
        
        # 计算word level qe: xcomet
        word_qe_scores = []
        if WORD_QE_MODE in ["only", "add"]:
            scores = score_word_level(word_qe_data)
            word_qe_scores.extend(scores)
        
        for i, item in enumerate(triplet_list):
            j = item["index"]
            if WORD_QE_MODE in ["only", "add"]: # word level qe exists
                if compute_val_reward:
                    final_scores[j]['format_score'] = fmt
                    final_scores[j]['bleu_score'] = item['bleu_score'] / 100.0
                    final_scores[j]['comet_score'] = comet_scores[i]
                    final_scores[j]['word_level_qe'] = word_qe_scores[i]

                    final_scores[j]['score'] = final_scores[j]['format_score'] + final_scores[j]['bleu_score'] + final_scores[j]['comet_score'] # only时，不包括 word level qe，防止hacking
                    if WORD_QE_MODE == "add":
                        final_scores[j]['score'] += WORD_QE_WEIGHT * final_scores[j]['word_level_qe']
                   
                    print(f"Item {j}: Validation final={final_scores[j]['score']:.4f} "
                        f"format_score={fmt}, bleu_score={final_scores[j]['bleu_score']:.4f}, comet_score={final_scores[j]['comet_score']:.4f}, word_level_qe={final_scores[j]['word_level_qe']:.4f}")
                else: # 训练阶段
                    if WORD_QE_MODE == "only":
                        final_scores[j] = fmt + word_qe_scores[i]
                        print(f"Item {j}: final={final_scores[j]:.4f} "
                            f"seq: format={fmt}; "
                            f"wordQE_mode={WORD_QE_MODE}, wordQE={word_qe_scores[i]:.4f})")
                    else: # WORD_QE_MODE == "add"
                        final_scores[j] = fmt + item['bleu_score'] / 100.0 + comet_scores[i] + WORD_QE_WEIGHT * word_qe_scores[i]
                        print(f"Item {j}: final={final_scores[j]:.4f} "
                            f"(seq: format={fmt}, bleu={item['bleu_score']:.4f}, comet={comet_scores[i]:.4f}; "
                            f"wordQE_mode={WORD_QE_MODE}, wordQE={word_qe_scores[i]:.4f})")

            else: # if WORD_QE_MODE == "off":
                if compute_val_reward:
                    final_scores[j]['format_score'] = fmt
                    final_scores[j]['bleu_score'] = item['bleu_score'] / 100.0
                    final_scores[j]['comet_score'] = comet_scores[i]
                    final_scores[j]['score'] = final_scores[j]['format_score'] + final_scores[j]['bleu_score'] + final_scores[j]['comet_score']
                    print(f"Item {j}: Validation final={final_scores[j]['score']:.4f} "
                        f"format_score={fmt}, bleu_score={final_scores[j]['bleu_score']:.4f}, comet_score={final_scores[j]['comet_score']:.4f}")
                else: # 训练阶段
                    final_scores[j] = fmt + item['bleu_score'] / 100.0 + comet_scores[i]
                    print(f"Item {j}: final={final_scores[j]:.4f} "
                        f"(seq: format={fmt}, bleu={item['bleu_score']:.4f}, comet={comet_scores[i]:.4f}; "
                        f"wordQE_mode={WORD_QE_MODE}, wordQE={word_qe_scores[i]:.4f})")

    print(f"Batch processing completed: {len(final_scores)} scores computed")
    return final_scores

# =========================================================
#           统一入口：兼容 Naive / DAPO / Batch 调用
# =========================================================
def compute_score(*args, **kwargs):
    """
    兼容三种调用：
    1) 批量（BatchRewardManager 风格）：
       compute_score(data_sources=[], solution_strs=[], ground_truths=[], extra_infos=None, ...)
    2) 单条（位置参数）：
       compute_score(data_source, solution_str, ground_truth, extra_info=None)
    3) 单条（关键字参数，DAPO 风格）：
       compute_score(data_source=..., solution_str=..., ground_truth=..., extra_info=None)
    """
    # 批量
    if 'data_sources' in kwargs or 'solution_strs' in kwargs or 'ground_truths' in kwargs:
        return compute_score_batch(
            kwargs.get('data_sources', []),
            kwargs.get('solution_strs', []),
            kwargs.get('ground_truths', []),
            kwargs.get('extra_infos', None),
            kwargs.get('compute_val_reward', False),
            kwargs.get('micro_batch_size', 8),
        )
    # 单条（位置参数）
    if len(args) >= 3:
        return compute_score_single(
            args[0], args[1], args[2],
            args[3] if len(args) > 3 else kwargs.get('extra_info', None),
            args[4] if len(args) > 4 else kwargs.get('compute_val_reward', False)
        )
    # 单条（关键字参数）
    if {'data_source', 'solution_str', 'ground_truth'} <= set(kwargs.keys()):
        return compute_score_single(
            kwargs['data_source'], 
            kwargs['solution_str'], 
            kwargs['ground_truth'], 
            kwargs.get('extra_info', None),
            kwargs.get('compute_val_reward', False)
        )
    raise ValueError(f"Invalid arguments for compute_score: args={args}, kwargs={kwargs}")