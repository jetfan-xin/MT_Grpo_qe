# my_comet_reward.py  
from comet import download_model, load_from_checkpoint  

import re
import logging
loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in loggers:
    logger.setLevel(logging.WARNING)
  
# 全局变量缓存模型  
_comet_model = None  
  
def _load_comet_model():  
    global _comet_model  
    if _comet_model is None:  
        print("Loading COMET model...")
        _comet_model = load_from_checkpoint("/mnt/workspace/xintong/pjh/models/wmt23-cometkiwi-da-xl/checkpoints/model.ckpt")  
    return _comet_model  

def compute_bleu(lg_pair, ref, pred):  
    import sacrebleu  
    import re  
      
    pred = pred if isinstance(pred, str) else ""  
    tgt_lang = lg_pair.split("-")[1]  
    tokenize = "zh" if tgt_lang == "zh" else "ja-mecab" if tgt_lang == "ja" else "13a"  
      
    bleu = sacrebleu.sentence_bleu(pred, [ref], lowercase=True, tokenize=tokenize)
    return float(bleu.score)

# def format_reward(solution_str: str) -> float:  
#     # 检查您期望的格式，例如：  
#     # - 是否包含特定标记  
#     # - 是否遵循特定结构  
#     # - 长度是否合理等  
#     pattern = r'^<think>.*?</think>\s*<translate>.*?</translate>(?![\s\S])'
#     match = re.match(pattern, solution_str, re.DOTALL | re.MULTILINE)
#     if match:  
#         return 1.0  
#     else:  
#         return 0.0  
def extract_solution(solution_str: str) -> str:
    """Extracts the final answer from the model's response string.
    
    Args:
        solution_str: Raw response string from the language model
        
    Returns:
        Tuple containing (extracted_answer, processed_string)
    """
    # # Split response to isolate assistant output
    # if "Assistant:" in solution_str: # base
    #     processed_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str: # qwen and tower
    #     processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # elif "<|start_header_id|>assistant<|end_header_id|>" in solution_str: # llama3
    #     processed_str = solution_str.split("<|start_header_id|>assistant<|end_header_id|>", 1)[1]
    # else:
    #     print("[Error] Failed to locate model response header")
        # return None, solution_str
# 
    # Extract final answer using XML-style tags
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
            print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
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


def compute_score(data_source, solution_str, ground_truth, extra_info=None):  
    # 从extra_info获取必要信息  
    lg_pair = extra_info.get("lg", "en-zh") if extra_info else "en-zh"  
    src_text = extra_info.get("source", ground_truth) if extra_info else ground_truth  
      
    format_score = validate_response_structure(solution_str)
      
    if not format_score:  
        print("invalid format")
        return -2.0  # 格式错误惩罚  
      
    # 计算BLEU分数  
    # match = re.search(r'<translate>(.*?)</translate>', solution_str, re.DOTALL)
    answer_text = extract_solution(solution_str)
    if answer_text is  None:
        print("format score is 1.0 but no <translate> tag found in completion: ", solution_str)
        return -2.0

    bleu_score = compute_bleu(lg_pair, ground_truth, answer_text)  
      
    # 计算COMET分数  
    model = _load_comet_model()  
    # comet_data = [{"src": src_text, "mt": answer_text, "ref": ground_truth}]  
    comet_data = [{"src": src_text, "mt": answer_text}]  
    comet_scores = model.predict(comet_data, batch_size=8, gpus=0, progress_bar=False).scores  
    comet_score = comet_scores[0]  
    # print("comet score: ", comet_score, "bleu score: ", bleu_score, "format score: ", format_score)
      
    # Merge: 组合BLEU和COMET分数  
    # 使用连续分数，缩放到合理范围  
    final_score = format_score + (bleu_score / 100.0) + (comet_score)  
    print("final score: ", final_score)
      
    return final_score