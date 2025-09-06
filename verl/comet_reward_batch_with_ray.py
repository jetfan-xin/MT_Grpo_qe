# COMET on a dedicated Ray GPU actor. 当所有GPU用于训练，在GRPO ADVANTAGE ESTIMATION - Starting computation时会卡住


import re, os
import logging
from typing import List, Dict, Any, Optional

import ray
import torch
from tqdm import tqdm
# ---------- Logging: 降低外部包日志噪音 ----------
for name in logging.root.manager.loggerDict:
    try:
        logging.getLogger(name).setLevel(logging.WARNING)
    except Exception:
        pass

# ---------- 配置项（可用环境变量覆盖） ----------
_COMET_CKPT = os.getenv(
    "COMET_CKPT",
    "/ltstorage/home/4xin/.cache/huggingface/hub/"
    "models--Unbabel--wmt23-cometkiwi-da-xl/"
    "snapshots/33858b2239a139d497d9c74952c88b89a8c06213/"
    "checkpoints/model.ckpt",
)
_COMET_ACTOR_NAME = os.getenv("COMET_ACTOR_NAME", "comet_actor")
_COMET_BATCH = int(os.getenv("COMET_BATCH", "32"))  # 批量预测的 batch size
_COMET_NAMESPACE = os.getenv("COMET_NAMESPACE", "verl_comet")
_COMET_ACTOR = None  # 全局缓存

## =====导入legacy===== ##
import importlib.util, sys
def load_legacy_as(alias_name: str, legacy_root: str):
    pkg_init = os.path.join(legacy_root, "comet", "__init__.py")
    spec = importlib.util.spec_from_file_location(
        alias_name, pkg_init,
        submodule_search_locations=[os.path.dirname(pkg_init)]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias_name] = mod
    spec.loader.exec_module(mod)  # type: ignore
    return mod

LEGACY_ROOT = os.path.expanduser("~/MT_Grpo_qe/wmt22-comet-legacy")
comet_legacy = load_legacy_as("comet_legacy", LEGACY_ROOT)

# 新栈（sequence-level, unified_metric）
from comet.models import load_from_checkpoint as load_ckpt_new  # 新栈（sequence-level, unified_metric） # 本地 ckpt，避免下载
# 旧栈（word-level, unite_metric_multi_task）
import importlib
comet_legacy_models = importlib.import_module("comet_legacy.models")  # 关键！
load_ckpt_legacy = getattr(comet_legacy_models, "load_from_checkpoint")
# load_ckpt_legacy = comet_legacy.models.load_from_checkpoint
# 词级（旧栈）：
try:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    word_model = load_ckpt_legacy("/ltstorage/home/4xin/MT_Grpo_qe/ckpts/comet/WMT24-QE-task2-baseline/checkpoints/model.fixed.ckpt").to(device).eval()
    print("成功加载QE checkpoint")
except:
    print("加载QE checkpoint失败")

# =========================================================
#                Ray Actor：独占 1 张 GPU
# =========================================================
@ray.remote(num_gpus=1)
class CometActor:
    def __init__(self, ckpt_path: str):
        cvd = os.getenv("CUDA_VISIBLE_DEVICES")
        print(f"[COMET-Actor] CVD={cvd}  cuda.is_available={torch.cuda.is_available()}  count={torch.cuda.device_count()}")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # 序列级（新栈）
        self.model = load_ckpt_new(ckpt_path).to(self.device)
        self.model.eval()
        print(f"[COMET-Actor] model loaded on {self.device}")

    def predict(self, data: List[Dict[str, str]], batch_size: int = 32):
        """对外提供统一的 predict；内部自动选 GPU/CPU 接口。"""
        # 新版 COMET (Lightning 2.x) 用 accelerator/devices；旧版用 gpus=...
        if self.device.startswith("cuda"):
            try:
                return self.model.predict(
                    data, batch_size=batch_size, progress_bar=False,
                    accelerator="gpu", devices=1
                )
            except TypeError:
                return self.model.predict(
                    data, batch_size=batch_size, progress_bar=False, gpus=1
                )
        else:
            try:
                return self.model.predict(
                    data, batch_size=batch_size, progress_bar=False,
                    accelerator="cpu", devices=1
                )
            except TypeError:
                return self.model.predict(
                    data, batch_size=batch_size, progress_bar=False, gpus=0
                )

# 全局 actor 句柄
_COMET_ACTOR = None

def _ray_ensure_initialized():
    if ray.is_initialized():
        return
    try:
        ray.init(address="auto", ignore_reinit_error=True, namespace=_COMET_NAMESPACE)
    except Exception:
        ray.init(ignore_reinit_error=True, namespace=_COMET_NAMESPACE)


def _get_comet_actor():
    """get-or-create（带 namespace）"""
    global _COMET_ACTOR
    _ray_ensure_initialized()
    if _COMET_ACTOR is not None:
        return _COMET_ACTOR
    try:
        _COMET_ACTOR = ray.get_actor(_COMET_ACTOR_NAME, namespace=_COMET_NAMESPACE)
    except Exception:
        _COMET_ACTOR = CometActor.options(
            name=_COMET_ACTOR_NAME, lifetime="detached", namespace=_COMET_NAMESPACE
        ).remote(_COMET_CKPT)
    return _COMET_ACTOR


def _predict_actor(data, batch_size: int):
    """稳定预测：失败时先重试 get，不再盲目“重建同名”"""
    actor = _get_comet_actor()
    try:
        return ray.get(actor.predict.remote(data, batch_size=batch_size))
    except Exception as e:
        msg = str(e)
        print("[COMET] predict failed; try reusing existing actor:", msg[:200])
        # 尝试重新获取句柄（可能之前缓存失效）
        try:
            actor2 = ray.get_actor(_COMET_ACTOR_NAME, namespace=_COMET_NAMESPACE)
            return ray.get(actor2.predict.remote(data, batch_size=batch_size))
        except Exception as e2:
            print("[COMET] reuse also failed; creating a TEMP anonymous actor:", str(e2)[:200])
            # 创建匿名（不命名）actor，避免 “already taken”
            temp_actor = CometActor.options(namespace=_COMET_NAMESPACE, lifetime="detached").remote(_COMET_CKPT)
            return ray.get(temp_actor.predict.remote(data, batch_size=batch_size))

# =========================================================
#                     你原有的工具函数
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
        print("[Error] No valid <translate> tags found")
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

# =========================================================
#                   单条评分（Naive / DAPO 单样本）
# =========================================================
def compute_score_single(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None,
) -> float:
    lg_pair = extra_info.get("lg", "en-zh") if extra_info else "en-zh"
    src_text = extra_info.get("source", ground_truth) if extra_info else ground_truth

    format_score = validate_response_structure(solution_str)
    if not format_score:
        print("invalid format")
        return -3.0

    ans = extract_solution(solution_str)
    if ans is None:
        print("format score is 1.0 but no <translate> tag found in completion")
        return -3.0

    bleu_score = compute_bleu(lg_pair, ground_truth, ans)

    comet_data = [{"src": src_text, "mt": ans}]
    out = _predict_actor(comet_data, batch_size=8)  # ← 走 Ray Actor（GPU）
    scores = out["scores"] if isinstance(out, dict) and "scores" in out else out.scores
    comet_score = float(scores[0])

    final_score = float(format_score) + (bleu_score / 100.0) + comet_score
    print("final score:", final_score)
    return final_score

# =========================================================
#                   批量评分（BatchRewardManager）
# =========================================================
def compute_score_batch(
    data_sources: List[str],
    solution_strs: List[str],
    ground_truths: List[str],
    extra_infos: Optional[List[Optional[Dict[str, Any]]]] = None,
    micro_batch_size: int = 8,  # 未使用（一次性送 actor，actor 内部再 batch）
) -> List[float]:
    if extra_infos is None:
        extra_infos = [None] * len(solution_strs)

    triplet_list = []
    final_scores: List[float] = []
    invalid_items: List[int] = []

    print(f"Processing batch of {len(solution_strs)} items...")
    print("data_sources", len(data_sources),
          "solution_strs", len(solution_strs),
          "ground_truths", len(ground_truths),
          "extra_infos", len(extra_infos))

    for i in tqdm(range(len(solution_strs)), desc="checking format and building triplets"):
        sol = solution_strs[i]
        gt = ground_truths[i]
        info = extra_infos[i]
        lg_pair = info.get("lg", "en-zh") if info else "en-zh"
        src_text = info.get("source", gt) if info else gt

        if not validate_response_structure(sol):
            invalid_items.append(i)
            final_scores.append(-3.0)
            continue

        ans = extract_solution(sol)
        if ans is None:
            invalid_items.append(i)
            final_scores.append(-3.0)
            continue

        bleu = compute_bleu(lg_pair, gt, ans)
        triplet_list.append({
            "triplet": {"src": src_text, "mt": ans},
            "format_score": True,
            "bleu_score": bleu,
            "index": i,
        })

    print(f"invalid items number {len(invalid_items)} / {len(solution_strs)}")

    if triplet_list:
        comet_triplets = [x["triplet"] for x in triplet_list]
        print("Processing comet triplets", len(comet_triplets), comet_triplets[:2])

        # 一次性给 actor；actor 内部会在 GPU 上以 _COMET_BATCH 做 batching
        out = _predict_actor(comet_triplets, batch_size=_COMET_BATCH)
        scores = out["scores"] if isinstance(out, dict) and "scores" in out else out.scores
        comet_scores = [float(s) for s in scores]

        for i, item in enumerate(triplet_list):
            j = item["index"]
            fmt = 1.0  # 上面已验证结构
            bleu = item["bleu_score"]
            comet = comet_scores[i]
            score = fmt + (bleu / 100.0) + comet
            # 回填到原索引
            while len(final_scores) <= j:
                final_scores.append(0.0)
            final_scores[j] = score
            print(f"Item {j}: final={score:.4f} (format=1.0, bleu={bleu:.2f}, comet={comet:.4f})")

    # 补齐长度（极端情况下）
    while len(final_scores) < len(solution_strs):
        final_scores.append(-3.0)

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
            kwargs.get('micro_batch_size', 8),
        )
    # 单条（位置参数）
    if len(args) >= 3:
        return compute_score_single(
            args[0], args[1], args[2],
            args[3] if len(args) > 3 else kwargs.get('extra_info', None)
        )
    # 单条（关键字参数）
    if {'data_source', 'solution_str', 'ground_truth'} <= set(kwargs.keys()):
        return compute_score_single(
            kwargs['data_source'], kwargs['solution_str'], kwargs['ground_truth'], kwargs.get('extra_info', None)
        )
    raise ValueError(f"Invalid arguments for compute_score: args={args}, kwargs={kwargs}")