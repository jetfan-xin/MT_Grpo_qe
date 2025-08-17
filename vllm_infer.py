from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import json
import tqdm
from pathlib import Path
import os

lang_map = {
    "en": "English",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    'de': "German",
    'fr': "French",
    'it': "Italian",
    'th': "Thai",
    'ru': "Russian",
    'pt': "Portuguese",
    'es': "Spanish",
    'hi': "Hindi",
    'tr': "Turkish",
    'ar': "Arabic",
}
def make_prefix(user_input, template_type, src_lang_name, tgt_lang_name):
    if template_type == 'base':
        prefix = f"""A conversation between User and Assistant. The User asks for a translation from {src_lang_name} to {tgt_lang_name}, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the final translation. The reasoning process and final translation are enclosed within <think> </think> and <translate> </translate> tags, respectively, i.e., <think> reasoning process here </think><translate> final translation here </translate>. \n\nUser:{user_input}\nAssistant:"""
        # prefix = f"You are a helpful translation assistant. Translate the following text from {src_lang_name} to {tgt_lang_name}. The translation should be accurate and fluent. Only provide the translation without any additional comments or explanations. \n\n{src_lang_name}: {user_input}\n{tgt_lang_name}:"
        return prefix
    elif template_type == 'chat':
        messages = [
        {"role": "system", "content": f"You are a helpful translation assistant. There is a conversation between User and Assistant. The user asks for a translation from {src_lang_name} to {tgt_lang_name}, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the final translation. The reasoning process and final translation are enclosed within <think> </think> and <translate> </translate> tags, respectively, i.e., <think> reasoning process here </think><translate> final translation here </translate>."},
        {"role": "user", "content": user_input}
                ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return text + "<think>"

def translate(data_file, lang, template_type, batch_size=8):
    with open(data_file, "r") as f:
        data = [json.loads(line) for line in f.readlines()]
    src_lang,tgt_lang = lang.split('2')
    result = []
    
    # 准备所有的输入文本
    all_texts = []
    all_items = []
    for item in data:
        src = item[src_lang]
        tgt = item[tgt_lang]
        text = make_prefix(src, template_type, lang_map[src_lang], lang_map[tgt_lang])
        all_texts.append(text)
        all_items.append({"src": src, "tgt": tgt})
    
    # Batch推理
    for i in tqdm.tqdm(range(0, len(all_texts), batch_size), desc="Processing batches"):
        batch_texts = all_texts[i:i+batch_size]
        batch_items = all_items[i:i+batch_size]
        
        # 批量生成
        batch_results = llm.generate(batch_texts, sampling_params)
        
        # 处理批量结果
        for j, (res, item) in enumerate(zip(batch_results, batch_items)):
            # 从RequestOutput对象中提取生成的文本
            generated_text = res.outputs[0].text
            result.append({
                "idx": i + j, 
                src_lang: item["src"], 
                tgt_lang: item["tgt"], 
                "mt": generated_text
            })
    
    return result

if __name__ == "__main__":
    # Initialize the tokenizer

    # Configurae the sampling parameters (for thinking mode)
    TEMPERATURE=0.2
    TOP_P=0.95
    MAX_TOKENS=2048
    BATCH_SIZE=16  # 降低batch size以匹配工作版本并减少内存使用
    sampling_params = SamplingParams(
        temperature=TEMPERATURE, 
        top_p=TOP_P, 
        max_tokens=MAX_TOKENS,
        repetition_penalty=1.1,
        skip_special_tokens=False  # 添加这个参数以保持与工作版本一致
    )

    # Initialize the vLLM engine
    model_path = "/mnt/workspace/xintong/pjh/All_result/mt_grpo/verl_grpo_xwang/merge_model/qwen2.5_3b_gtpo_bleu_comet_entropy_b1"
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,  # 降低内存利用率
        max_model_len=16384,  # 添加最大模型长度参数
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    data_folder = "data/test/json"
    
    for file in Path(data_folder).glob("*.jsonl"):
        data_file = str(file)
        if "dezh" in data_file:
            lang = "de2zh"
        elif "zhen" in data_file:
            lang = "zh2en"
        elif "enzh" in data_file:
            lang = "en2zh"
        elif "enja" in data_file:
            lang = "en2ja"
        elif "deen" in data_file:
            lang="de2en"
        else:
            print(f"Unsupported language in file: {data_file}")
            continue

        # if lang == "zh2en" or lang=="en2zh":
        #     continue
        print(f"Processing file: {data_file}", lang)
        # Parse the file and translate
        save_name = data_file.split("/")[-1].replace(".jsonl", f"_mt.json")
        save_path = "/mnt/workspace/xintong/pjh/All_result/mt_grpo/verl_grpo_result/qwen2.5_3b_gtpo_bleu_comet_entropy_b1_rp/"
        os.makedirs(save_path, exist_ok=True)

        result = translate(data_file, lang, template_type='base', batch_size=BATCH_SIZE)    
        # result = translate(data_file, lang, template_type='chat', batch_size=BATCH_SIZE)    
    

        json.dump(result, open(save_path + save_name, "w", encoding="utf-8"), ensure_ascii=False, indent=4)
