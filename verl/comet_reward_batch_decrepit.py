# my_comet_reward.py
import re, os
import logging
loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in loggers:
    logger.setLevel(logging.WARNING)
from tqdm import tqdm
import torch
import ray
from comet.models import load_from_checkpoint as load_ckpt_new # 用于load新栈（sequence-level, unified_metric）

_COMET_CKPT = os.getenv(
    "COMET_CKPT",
    "/ltstorage/home/4xin/.cache/huggingface/hub/"
    "models--Unbabel--wmt23-cometkiwi-da-xl/"
    "snapshots/33858b2239a139d497d9c74952c88b89a8c06213/"
    "checkpoints/model.ckpt",
)

## =====导入legacy===== ##
import importlib
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

# 旧栈（word-level, unite_metric_multi_task）
comet_legacy_models = importlib.import_module("comet_legacy.models")  # 关键！
load_ckpt_legacy = getattr(comet_legacy_models, "load_from_checkpoint")

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
# 启用Ray调度
@ray.remote(num_gpus=1) 
def _score_batch_on_gpu(comet_ckpt, triplets, batch_size, gpus):
    import torch
    model = load_ckpt_new(comet_ckpt).to("cuda") # 新栈（sequence-level, unified_metric）
    return model.predict(triplets, batch_size=batch_size, gpus=gpus, progress_bar=False)

# 全局变量缓存模型  
_comet_model = None

def _load_comet_model():  
    global _comet_model  
    if _comet_model is None:  
        print("Loading COMET model...")
        
        # 在 Ray 多进程环境中重新初始化 CUDA 上下文
        # 此处未调用 torch.cuda.init() —— PyTorch 没这个公共 API
        # 在不可用时不硬转 cuda
        try:
            # 尝试重新初始化 CUDA 环境
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            torch.cuda.init()
        except Exception as e:
            print(f"CUDA initialization warning: {e}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"CUDA available: {torch.cuda.is_available()} | using device={device}")

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            gpu_memory = torch.cuda.get_device_properties(current_device).total_memory
            print(f"Available GPUs: {gpu_count}, Current device: {current_device}, GPU memory: {gpu_memory / 1e9:.1f}GB")
            
            # Load COMET model with device specification
            _comet_model = load_ckpt_new(_COMET_CKPT)
            # Move to specific GPU if available
            # Use a different GPU if multiple GPUs available to avoid conflict with vLLM
            target_device = f"cuda:{(current_device + 1) % gpu_count}" if gpu_count > 1 else f"cuda:{current_device}"
            _comet_model = _comet_model.to(target_device)
            print(f"Loaded COMET model on {target_device}")
        else:
            print(f"CUDA is not available: {torch.cuda.is_available()}")
            print("Loading COMET model on CPU (this will be slower)")
            _comet_model = load_ckpt_new(_COMET_CKPT).to(device).eval()

    return _comet_model  

# =========================================================
#                     原有的工具函数：没改
# =========================================================
def compute_bleu(lg_pair, ref, pred):  
    import sacrebleu
    pred = pred if isinstance(pred, str) else ""  
    tgt_lang = lg_pair.split("-")[1]  
    tokenize = "zh" if tgt_lang == "zh" else "ja-mecab" if tgt_lang == "ja" else "13a"  
    bleu = sacrebleu.sentence_bleu(pred, [ref], lowercase=True, tokenize=tokenize)
    return float(bleu.score)


def extract_solution(solution_str: str) -> str:
    """Extracts the final answer from the model's response string.
    
    Args:
        solution_str: Raw response string from the language model
        
    Returns:
        Tuple containing (extracted_answer, processed_string)
    """
    answer_pattern = r'<translate>(.*?)</translate>'
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))
    if not matches:
        print("[Error] No valid answer tags found")
        return None
    final_answer = matches[-1].group(1).strip()
    return final_answer


def validate_response_structure(processed_str: str) -> bool:
    """Performs comprehensive validation of response structure.
    
    Args:
        processed_str: Processed response string from the model
        
    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    # print("\n[Structure Validation]")
    validation_passed = True

    # Check required tags
    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1),
        'answer_start': ('<translate>', 1),
        'answer_end': ('</translate>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)
        # print(f"  {tag_str}: count={count}, position={pos}")
        if count != expected_count:
            # print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False
    # Verify tag order
    if (positions['think_start'] > positions['think_end'] or
        positions['think_end'] > positions['answer_start'] or
        positions['answer_start'] > positions['answer_end']):
        # print("  [Error] Incorrect tag order: Expected <think>...</think><answer>...</answer>")
        validation_passed = False
    # else:
    #     print("  Tag sequence validation passed")
    return validation_passed

# =========================================================
#                          单条评分：没改
# =========================================================
def compute_score_single(data_source, solution_str, ground_truth, extra_info=None):  
    """
    Single-item version of compute_score function for backward compatibility.
    Used by NaiveRewardManager.
    """
    lg_pair = extra_info.get("lg", "en-zh") if extra_info else "en-zh"  
    src_text = extra_info.get("source", ground_truth) if extra_info else ground_truth  
    
    format_score = validate_response_structure(solution_str)
    
    if not format_score:  
        print("invalid format")
        return -3.0  # 格式错误惩罚，与batch版本保持一致  
    
    answer_text = extract_solution(solution_str)
    if answer_text is  None:
        print("format score is 1.0 but no <translate> tag found in completion: ", solution_str)
        return -3.0

    bleu_score = compute_bleu(lg_pair, ground_truth, answer_text)  
    
    model = _load_comet_model()  
    comet_data = [{"src": src_text, "mt": answer_text}]  
    gpus_flag = 1 if torch.cuda.is_available() else 0 # <- 修复：控制是否分配gpu
    comet_scores = model.predict(comet_data, batch_size=8, gpus=gpus_flag, progress_bar=False).scores  
    comet_score = float(comet_scores[0])  # 直接用原始分数
    final_score = format_score + (bleu_score / 100.0) + comet_score  # BLEU缩放，COMET不缩放
    print("final score: ", final_score)
    return final_score

# =========================================================
#                   批量评分（BatchRewardManager）：改了
# =========================================================
def compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos=None, micro_batch_size=8):
    """
    Batch version of compute_score function.
    Migrated and optimized from MT-R1-Zero DataParallelCOMET.compute_comet_rm
    
    Args:
        data_sources: List of data sources
        solution_strs: List of solution strings
        ground_truths: List of ground truth strings
        extra_infos: List of extra info dicts (optional)
        micro_batch_size: Size of micro batches for COMET processing
        
    Returns:
        List of final scores
    """
    if extra_infos is None:
        extra_infos = [None] * len(solution_strs)
    
    triplet_list = []
    final_scores = []
    
    print(f"Processing batch of {len(solution_strs)} items...")
    print("data_sources", len(data_sources), "solution_strs", len(solution_strs),
          "ground_truths", len(ground_truths), "extra_infos", len(extra_infos))

    model = _load_comet_model()
    
    invalid_items=[]
    for i in tqdm(range(len(solution_strs)), desc="checking format and building triplets"):
        data_source = data_sources[i]
        solution_str = solution_strs[i]
        ground_truth = ground_truths[i]
        extra_info = extra_infos[i]
        
        lg_pair = extra_info.get("lg", "en-zh") if extra_info else "en-zh"
        src_text = extra_info.get("source", ground_truth) if extra_info else ground_truth
        
        format_score = validate_response_structure(solution_str)
        if not format_score:
            invalid_items.append(i)
            final_scores.append(-3.0)
            continue
        
        answer_text = extract_solution(solution_str)
        if answer_text is None:
            invalid_items.append(i)
            final_scores.append(-3.0)
            continue
        
        bleu_score = compute_bleu(lg_pair, ground_truth, answer_text)
        
        triplet_item = {"src": src_text, "mt": answer_text}
        triplet_list.append({
            "triplet": triplet_item,
            "format_score": format_score,
            "bleu_score": bleu_score,
            "index": i
        })
    print(f"invalid items number {len(invalid_items)} / {len(solution_strs)}")
    
    if triplet_list:
        comet_triplets = [item["triplet"] for item in triplet_list]
        print("Processing comet triplets", len(comet_triplets), comet_triplets[:2])

        comet_scores_flat = []

        # 直接用原始分数
        gpus_flag = 1 if torch.cuda.is_available() else 0 # <- 修复：控制是否分配gpu
        scores = ray.get(_score_batch_on_gpu.remote(_COMET_CKPT, comet_triplets, batch_size=32, gpus=gpus_flag)) # 使用Ray分配一张卡
        comet_scores_flat.extend([float(score) for score in scores.scores])
        
        for i, item in enumerate(triplet_list):
            original_index = item["index"]
            format_score = item["format_score"]
            bleu_score = item["bleu_score"]
            comet_score = comet_scores_flat[i]  # 不再/100.0
            final_score = format_score + (bleu_score / 100.0) + comet_score
            while len(final_scores) <= original_index:
                final_scores.append(0.0)
            final_scores[original_index] = final_score
            print(f"Item {original_index}: final_score={final_score} (format={format_score}, bleu={bleu_score}, comet={comet_score})")
    while len(final_scores) < len(solution_strs):
        final_scores.append(-3.0)
    print(f"Batch processing completed: {len(final_scores)} scores computed")
    return final_scores

# =========================================================
#           统一入口：兼容 Naive / DAPO / Batch 调用
# =========================================================
def compute_score(*args, **kwargs):
    """
    Adaptive compute_score function that supports both single and batch processing.
    
    For single processing (NaiveRewardManager):
        compute_score(data_source, solution_str, ground_truth, extra_info=None)
        
    For batch processing (BatchRewardManager):
        compute_score(data_sources=[], solution_strs=[], ground_truths=[], extra_infos=None, ...)
    """
    # Check if this is a batch call (BatchRewardManager style)
    if 'data_sources' in kwargs or 'solution_strs' in kwargs or 'ground_truths' in kwargs:
        # Batch processing call
        data_sources = kwargs.get('data_sources', [])
        solution_strs = kwargs.get('solution_strs', [])
        ground_truths = kwargs.get('ground_truths', [])
        extra_infos = kwargs.get('extra_infos', None)
        micro_batch_size = kwargs.get('micro_batch_size', 8)
        
        print(f"Using BATCH processing for {len(solution_strs)} items")
        return compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos, micro_batch_size)
    
    # Check if this is positional arguments (single processing)
    elif len(args) >= 3:
        # Single processing call
        print("Using SINGLE processing for 1 item")
        data_source = args[0]
        solution_str = args[1] 
        ground_truth = args[2]
        extra_info = args[3] if len(args) > 3 else kwargs.get('extra_info', None)
        
        return compute_score_single(data_source, solution_str, ground_truth, extra_info)
    
    # Check if this is single item with keyword arguments (DAPO style)
    elif 'data_source' in kwargs and 'solution_str' in kwargs and 'ground_truth' in kwargs:
        # Single item with keyword arguments
        print("Using SINGLE processing for 1 item (keyword args)")
        data_source = kwargs['data_source']
        solution_str = kwargs['solution_str']
        ground_truth = kwargs['ground_truth']
        extra_info = kwargs.get('extra_info', None)
        
        return compute_score_single(data_source, solution_str, ground_truth, extra_info)
    
    else:
        raise ValueError(f"Invalid arguments for compute_score: args={args}, kwargs={kwargs}")
