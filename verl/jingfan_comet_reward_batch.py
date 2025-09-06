# my_comet_reward.py  
from comet import download_model, load_from_checkpoint

import re
import logging
loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in loggers:
    logger.setLevel(logging.WARNING)

from tqdm import tqdm
import torch
import contextlib #增加
# 全局变量缓存模型  
_comet_model = None  
  
def _load_comet_model():  
    global _comet_model  
    if _comet_model is None:  
        print("Loading COMET model...")
        
        # 确保CUDA环境正确设置
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            gpu_memory = torch.cuda.get_device_properties(current_device).total_memory
            print(f"Available GPUs: {gpu_count}, Current device: {current_device}, GPU memory: {gpu_memory / 1e9:.1f}GB")
            
            # Load COMET model with device specification
            try:
                # Try to load the downloaded model first
                print("Loading downloaded COMET model...")
                model_path = "/ltstorage/home/4xin/.cache/huggingface/hub/models--Unbabel--wmt23-cometkiwi-da-xl/snapshots/33858b2239a139d497d9c74952c88b89a8c06213"
                _comet_model = load_from_checkpoint(f"{model_path}/checkpoints/model.ckpt")
                print("Loaded downloaded COMET model successfully")
                
                # 设置模型为评估模式并使用bfloat16
                _comet_model.eval()
                _comet_model = _comet_model.to(dtype=torch.bfloat16)
            except Exception as e:
                print(f"Failed to load downloaded model: {e}")
                # Fall back to specific checkpoint
                try:
                    _comet_model = load_from_checkpoint("/mnt/workspace/xintong/pjh/models/wmt23-cometkiwi-da-xl/checkpoints/model.ckpt")
                    print("Loaded specific COMET checkpoint")
                except Exception as e2:
                    print(f"Failed to load specific checkpoint: {e2}")
                    raise Exception("Failed to load COMET model from both locations")
            
            # Smart GPU selection strategy: prioritize idle GPUs, then select best available
            print("Analyzing GPU availability...")
            
            # Collect GPU information
            gpu_info = []
            idle_gpus = []
            busy_gpus = []
            
            for device_id in range(gpu_count):
                try:
                    torch.cuda.set_device(device_id)
                    total_memory = torch.cuda.get_device_properties(device_id).total_memory
                    used_memory = torch.cuda.memory_allocated(device_id)
                    free_memory = total_memory - used_memory
                    memory_usage_ratio = used_memory / total_memory
                    gpu_name = torch.cuda.get_device_name(device_id)
                    
                    # Determine if GPU is idle (less than 1% memory used to be more conservative)
                    # Also check if GPU is actually free by looking at memory allocation
                    is_idle = memory_usage_ratio < 0.01 and used_memory < 1e9  # Less than 1GB used
                    
                    # Calculate safety score (how safe it is to use this GPU)
                    # Higher score = safer to use
                    safety_score = (free_memory / 1e9) * (1.0 - memory_usage_ratio)
                    
                    # GPU type bonus (A100 preferred)
                    gpu_type_bonus = 1.5 if "A100" in gpu_name else 1.0
                    
                    # Final score
                    score = safety_score * gpu_type_bonus
                    
                    gpu_info.append({
                        'device_id': device_id,
                        'name': gpu_name,
                        'total_memory': total_memory,
                        'used_memory': used_memory,
                        'free_memory': free_memory,
                        'memory_usage_ratio': memory_usage_ratio,
                        'is_idle': is_idle,
                        'safety_score': safety_score,
                        'score': score
                    })
                    
                    print(f"GPU {device_id} ({gpu_name}): "
                          f"{free_memory / 1e9:.1f}GB free, "
                          f"{memory_usage_ratio * 100:.1f}% used, "
                          f"idle: {is_idle}, "
                          f"safety: {safety_score:.1f}, "
                          f"score: {score:.1f}")
                    
                    if is_idle:
                        idle_gpus.append(device_id)
                    else:
                        busy_gpus.append(device_id)
                        
                except Exception as e:
                    print(f"Error checking GPU {device_id}: {e}")
            
            # Strategy 1: If we have idle GPUs, use the best idle GPU
            if idle_gpus:
                print(f"Found {len(idle_gpus)} idle GPUs: {idle_gpus}")
                # Select the best idle GPU
                best_idle_gpu = max(idle_gpus, key=lambda x: next(g for g in gpu_info if g['device_id'] == x)['score'])
                best_device = best_idle_gpu
                best_score = next(g for g in gpu_info if g['device_id'] == best_device)['score']
                print(f"Selected idle GPU {best_device} with score {best_score:.1f}")
            
            # Strategy 2: If no idle GPUs, select the safest busy GPU
            else:
                print("No idle GPUs found, selecting safest busy GPU...")
                # Sort by safety score (highest first)
                safe_gpus = sorted(gpu_info, key=lambda x: x['safety_score'], reverse=True)
                
                # Select the safest GPU that won't harm other processes
                # Avoid GPUs with >80% memory usage
                safe_candidates = [g for g in safe_gpus if g['memory_usage_ratio'] < 0.8]
                
                if safe_candidates:
                    best_device = safe_candidates[0]['device_id']
                    best_score = safe_candidates[0]['score']
                    print(f"Selected safest busy GPU {best_device} with safety score {safe_candidates[0]['safety_score']:.1f}")
                else:
                    # Fallback: select GPU with most free memory
                    best_device = max(gpu_info, key=lambda x: x['free_memory'])['device_id']
                    best_score = next(g for g in gpu_info if g['device_id'] == best_device)['score']
                    print(f"Fallback: selected GPU {best_device} with most free memory")
            
            # Additional safety check: if GPU 1 is heavily used, prefer other GPUs
            if best_device == 1 and gpu_count > 1:
                # Check if there are better alternatives
                alternative_gpus = [g for g in gpu_info if g['device_id'] != 1 and g['memory_usage_ratio'] < 0.5]
                if alternative_gpus:
                    best_alternative = max(alternative_gpus, key=lambda x: x['score'])
                    if best_alternative['score'] > best_score * 0.8:  # If alternative is at least 80% as good
                        best_device = best_alternative['device_id']
                        best_score = best_alternative['score']
                        print(f"Switched to safer alternative GPU {best_device} with score {best_score:.1f}")
            
            target_device = f"cuda:{best_device}"
            print(f"Final selection: {target_device} with score {best_score:.1f}")
            
            # Move model to selected GPU
            print(f"Moving model to {target_device}...")
            _comet_model = _comet_model.to(target_device)
            print(f"Model successfully moved to {target_device}")
            
            # Verify model is on correct device
            model_device = next(_comet_model.parameters()).device
            print(f"Model device verification: {model_device}")
            
            # Set model to evaluation mode
            _comet_model.eval()
            print("Model set to evaluation mode")
        else:
            print(f"CUDA is not available: {torch.cuda.is_available()}")
            print("Loading COMET model on CPU (this will be slower)")
            try:
                # Try to load the downloaded model first
                model_path = "/ltstorage/home/4xin/.cache/huggingface/hub/models--Unbabel--wmt23-cometkiwi-da-xl/snapshots/33858b2239a139d497d9c74952c88b89a8c06213"
                _comet_model = load_from_checkpoint(f"{model_path}/checkpoints/model.ckpt")
                print("Loaded downloaded COMET model on CPU")
            except Exception as e:
                print(f"Failed to load downloaded model: {e}")
                # Fall back to specific checkpoint
                try:
                    _comet_model = load_from_checkpoint("/mnt/workspace/xintong/pjh/models/wmt23-cometkiwi-da-xl/checkpoints/model.ckpt")
                    print("Loaded specific COMET checkpoint on CPU")
                except Exception as e2:
                    print(f"Failed to load specific checkpoint: {e2}")
                    raise Exception("Failed to load COMET model from both locations")
            _comet_model = _comet_model.to("cpu") 

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
        return -3.0  # 格式错误惩罚，与batch版本保持一致  
    
    answer_text = extract_solution(solution_str)
    if answer_text is  None:
        print("format score is 1.0 but no <translate> tag found in completion: ", solution_str)
        return -3.0

    bleu_score = compute_bleu(lg_pair, ground_truth, answer_text)  
    
    model = _load_comet_model()  
    comet_data = [{"src": src_text, "mt": answer_text}]  
    # Get the device the model is on
    model_device = next(model.parameters()).device
    gpu_id = model_device.index if model_device.type == 'cuda' else 0
    # comet_scores = model.predict(comet_data, batch_size=8, gpus=1, progress_bar=False).scores  
    # jingfan改为：
    use_gpu = next(model.parameters()).is_cuda
    ctx = torch.cuda.amp.autocast(enabled=False) if use_gpu else contextlib.nullcontext()
    with ctx:
        comet_scores = model.predict(
            comet_data,
            batch_size=8,
            gpus=1 if use_gpu else 0,
            progress_bar=False
        ).scores
    # /jingfan
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
    # comet_output = model.predict(micro_batch, batch_size=batch_size, gpus=1, progress_bar=False)
    # jingfan改为：
    use_gpu = next(model.parameters()).is_cuda
    ctx = torch.cuda.amp.autocast(enabled=False) if use_gpu else contextlib.nullcontext()
    with ctx:
        comet_output = model.predict(
            micro_batch,
            batch_size=batch_size,
            gpus=1 if use_gpu else 0,
            progress_bar=False
        )
    # /jingfan
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
        # Get the device the model is on
        model_device = next(model.parameters()).device
        gpu_id = model_device.index if model_device.type == 'cuda' else 0
        # scores = model.predict(comet_triplets, batch_size=32, gpus=1)
        # jingfan改为：
        use_gpu = next(model.parameters()).is_cuda
        ctx = torch.cuda.amp.autocast(enabled=False) if use_gpu else contextlib.nullcontext()
        with ctx:
            scores = model.predict(
                comet_triplets,
                batch_size=32,
                gpus=1 if use_gpu else 0
            )
        # /jaingfan
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