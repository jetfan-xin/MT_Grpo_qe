from openai import OpenAI
import base64
import json
import argparse
import datetime
import os
from pathlib import Path
import tqdm


openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

# models.list()返回一个模型列表，每个模型都有一个id属性
model_name = client.models.list().data[0].id

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def call_api(text, system_prompt):
    
    # base64_image = encode_image(image)
    response = client.chat.completions.create(
        # model="模型",
        model = model_name, # 图文
        messages=[
            {'role': 'system', 'content': system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                    ],
                }
        ],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=["zh2en", "en2zh"], required=True, help="Translation direction: zh2en or en2zh")
    args = parser.parse_args()

    if args.lang == "zh2en":
        system_prompt = "You are a helpful translation assistant. There is a conversation between User and Assistant. The user asks for a translation from Chinese to English, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the final translation. The reasoning process and final translation are enclosed within <think> </think> and <translate> </translate> tags, respectively, i.e., <think> reasoning process here </think><translate> final translation here </translate>."
        data_file = "data/wmt23_zhen.jsonl"
        source = "zh"
    elif args.lang == "en2zh":
        system_prompt = "You are a helpful translation assistant. There is a conversation between User and Assistant. The user asks for a translation from English to Chinese, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the final translation. The reasoning process and final translation are enclosed within <think> </think> and <translate> </translate> tags, respectively, i.e., <think> reasoning process here </think><translate> final translation here </translate>."
        data_file = "data/wmt24_enzh.jsonl"
        source = "en"

    with open(data_file, "r") as f:
        data = [json.loads(line) for line in f.readlines()]
    
    i = 0
    result = []
    for item in tqdm.tqdm(data):
        zh = item["zh"]
        en = item["en"]
        if source == "zh":
            res = call_api(zh, system_prompt)
        else:
            res = call_api(en, system_prompt)
        result.append({"idx": i, "zh": zh, "en": en, "mt": res})
        i += 1
    
    save_name = data_file.split("/")[-1].replace(".jsonl", f"_mt.json")
    save_path = '/mnt/workspace/xintong/pjh/All_result/mt_grpo/eval_qwen2.5-7b_grpo/'
    Path(save_path).mkdir(parents=True, exist_ok=True)
    json.dump(result, open(save_path + save_name, "w", encoding="utf-8"), ensure_ascii=False, indent=4)

