import argparse
import torch
import sys

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from PIL import Image
import os
import requests
import re
import json
import tqdm
from tqdm.contrib import tzip
from pathlib import Path
import random

def process_query(qs, sp=None):
    if sp is not None:
        messages = [
            {"role": "system", "content": sp},
            {
                "role": "user",
                "content": qs
            }
        ]
    else:
        messages = [
        {
            "role": "user",
            "content": qs,
        }
    ]
    # Preparation for inference
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return prompt

def generate(text, system_prompt):
    qs = text
    prompt = process_query(qs, system_prompt)
    # prompt = qs
    model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    model_inputs = model_inputs.to("cuda")

    # Inference: Generation of the output
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=4096,
            do_sample=True,
            num_beams=args.num_beams,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=151643
        )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    del model_inputs, generated_ids
    torch.cuda.empty_cache()
    return response

sp = None
# text_temp = "Translate the following English text into Chinese:"
# text_temp = "Translate the following Chinese text into English:"
# system_prompt = "You are a helpful translation assistant. Translate the following text from Chinese to English. The translation should be accurate and fluent. Only provide the translation without any additional comments or explanations. "
# system_prompt = "You are a helpful translation assistant. Translate the following text from {src_lang} to {tgt_lang}. The translation should be accurate and fluent. Only provide the translation without any additional comments or explanations. "
system_prompt = "You are a helpful translation assistant. There is a conversation between User and Assistant. The user asks for a translation from {src_lang} to {tgt_lang}, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the final translation. The reasoning process and final translation are enclosed within <think> </think> and <translate> </translate> tags, respectively, i.e., <think> reasoning process here </think><translate> final translation here </translate>."
# system_prompt=None
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

def translate(data_file, lang):
    with open(data_file, "r") as f:
        data = [json.loads(line) for line in f.readlines()]
    src_lang,tgt_lang = lang.split('2')
    i = 0
    result = []
    system_prompt_mt = system_prompt.format(src_lang=lang_map[src_lang], tgt_lang=lang_map[tgt_lang])
    for item in tqdm.tqdm(data):
        src = item[src_lang]
        tgt = item[tgt_lang]

        res = generate(src, system_prompt_mt)
        # res = system_prompt_mt + "\n<think> " + src + " </think><translate> " + tgt + " </translate>"

        result.append({"idx": i, src_lang: src, tgt_lang: tgt, "mt": res})
        i += 1
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model-path", type=str, default="/mnt/data/users/liamding/data/models/Qwen2.5-7B")
    # parser.add_argument("--model-path", type=str, default="/mnt/data/users/liamding/data/verl/qwen2_5_3b_function_rm_new/qwen2_5_3b_comet_bleu_rm")
    parser.add_argument("--model-path", type=str, default="/mnt/workspace/xintong/pjh/All_result/mt_grpo/verl_grpo_xwang/merge_model/qwen2.5_7b_r1-zero")
    
    
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.5)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    # parser.add_argument("--source_file", type=str, required=True)
    # parser.add_argument("--target_file", type=str, required=True)
    # parser.add_argument("--image_source", type=str, required=True)
    # parser.add_argument("--image_folder", type=str, required=True)
    # parser.add_argument("--prompt_temp", type=str, required=True)
    # parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="cuda:0",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    data_folder = "../data/test/json"
    
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


        print(f"Processing file: {data_file}", lang)
        # Parse the file and translate
        save_name = data_file.split("/")[-1].replace(".jsonl", f"_mt.json")
        result = translate(data_file, lang)    
    
        save_path = "/mnt/workspace/xintong/pjh/All_result/mt_grpo/verl_grpo_result/qwen2.5_7b_r1-zero_verl/"
        os.makedirs(save_path, exist_ok=True)

        json.dump(result, open(save_path + save_name, "w", encoding="utf-8"), ensure_ascii=False, indent=4)