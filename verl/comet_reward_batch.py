# my_comet_reward.py  
from comet import download_model, load_from_checkpoint  

import re
import logging
loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in loggers:
    logger.setLevel(logging.WARNING)

from tqdm import tqdm
import torch
  
# 全局变量缓存模型  
_comet_model = None  
  
def _load_comet_model():  
    global _comet_model  
    if _comet_model is None:  
        print("Loading COMET model...")
        
        # 在 Ray 多进程环境中重新初始化 CUDA 上下文
        try:
            # 尝试重新初始化 CUDA 环境
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            torch.cuda.init()
        except Exception as e:
            print(f"CUDA initialization warning: {e}")
        
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            gpu_memory = torch.cuda.get_device_properties(current_device).total_memory
            print(f"Available GPUs: {gpu_count}, Current device: {current_device}, GPU memory: {gpu_memory / 1e9:.1f}GB")
            
            # Load COMET model with device specification
            _comet_model = load_from_checkpoint("/mnt/workspace/xintong/pjh/models/wmt23-cometkiwi-da-xl/checkpoints/model.ckpt")  
            # Move to specific GPU if available
            # Use a different GPU if multiple GPUs available to avoid conflict with vLLM
            target_device = f"cuda:{(current_device + 1) % gpu_count}" if gpu_count > 1 else f"cuda:{current_device}"
            _comet_model = _comet_model.to(target_device)
            print(f"Loaded COMET model on {target_device}")
        else:
            print(f"CUDA is not available: {torch.cuda.is_available()}")
            print("Loading COMET model on CPU (this will be slower)")
            _comet_model = load_from_checkpoint("/mnt/workspace/xintong/pjh/models/wmt23-cometkiwi-da-xl/checkpoints/model.ckpt").to("cuda") 

    return _comet_model  

def compute_bleu(lg_pair, ref, pred):  
    import sacrebleu  
    import re  
      
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
        return -2.0  # 格式错误惩罚  
    
    answer_text = extract_solution(solution_str)
    if answer_text is  None:
        print("format score is 1.0 but no <translate> tag found in completion: ", solution_str)
        return -2.0

    bleu_score = compute_bleu(lg_pair, ground_truth, answer_text)  
    
    model = _load_comet_model()  
    comet_data = [{"src": src_text, "mt": answer_text}]  
    comet_scores = model.predict(comet_data, batch_size=8, gpus=1, progress_bar=False).scores  
    comet_score = float(comet_scores[0])  # 直接用原始分数
    final_score = format_score + (bleu_score / 100.0) + comet_score  # BLEU缩放，COMET不缩放
    print("final score: ", final_score)
    
    return final_score


def _forward_micro_batch(model, micro_batch):
    """
    Process a micro batch of triplets using COMET model.
    Migrated from MT-R1-Zero DataParallelCOMET._forward_micro_batch
    """
    batch_size = len(micro_batch)
    print(f"comet_reward.py forward micro_batch: {batch_size}")
    comet_output = model.predict(micro_batch, batch_size=batch_size, gpus=1, progress_bar=False)
    # 直接返回原始分数，不再*100
    scores = [float(score) for score in comet_output.scores]
    return scores

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
    
    else:
        raise ValueError(f"Invalid arguments for compute_score: args={args}, kwargs={kwargs}")

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
        scores = model.predict(comet_triplets, batch_size=32, gpus=1)
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