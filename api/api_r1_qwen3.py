# pip install openai==1.35.10
import datetime
import json
import openai
import time
import base64
import tqdm
from pathlib import Path
from PIL import Image
from io import BytesIO
import argparse

with open('/mnt/workspace/xintong/api_key.txt', 'r') as f:
    lines = f.readlines()
    
API_KEY = lines[0].strip()
BASE_URL = lines[1].strip()
openai.api_key = API_KEY
openai.base_url = BASE_URL


# text_temp = "You are a helpful translation assistant. There is a conversation between User and Assistant. The user asks for a translation from {src_lang} to {tgt_lang}, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the final translation. The final translation is enclosed within <translate> </translate> tags, i.e., <translate> final translation here </translate>.\n\{src_lang}:{src_text}\n"
text_temp = "Translate the following sentence from {src_lang} to {tgt_lang}. The final translation is enclosed within <translate> </translate> tags, i.e., <translate> final translation here </translate>.\n{src_lang}:{src_text}\n"

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


def call_qwen3_thinking(text):
    reasoning_content = ""  # 定义完整思考过程
    answer_content = ""     # 定义完整回复
    is_answering = False   # 判断是否结束思考过程并开始回复

    # 创建聊天完成请求
    completion = openai.chat.completions.create(
        model=model_name,  # 此处以 qwq-32b 为例，可按需更换模型名称
        extra_body={"enable_thinking": True},
        messages=[
            {"role": "user", "content": text}
        ],
        stream=True,
    )
    for chunk in completion:
        if not chunk.choices:  
            continue  # 跳过无效数据
        delta = chunk.choices[0].delta

        # 记录思考过程
        if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
            reasoning_content += delta.reasoning_content
        else:
            # 进入回复阶段
            if delta.content and not is_answering:
                is_answering = True

            # 记录最终的回答内容
            answer_content += delta.content
    
    return reasoning_content, answer_content

def call_r1(text):
    response = openai.chat.completions.create(
        model="deepseek-r1",  
        messages=[
            {"role": "user", "content": text}
        ],
    )

    return response.choices[0].message.reasoning_content, response.choices[0].message.content


def translate(data_file, lang):
    result = []
    with open(data_file, "r") as f:
        data = [json.loads(line) for line in f.readlines()]
    src_lang,tgt_lang = lang.split('2')

   
    sleep_times = [5, 10, 20, 40, 60]
    i = 0
    for item in tqdm.tqdm(data):
        src = item[src_lang]
        tgt = item[tgt_lang]
        text = text_temp.format(src_lang=lang_map[src_lang], tgt_lang=lang_map[tgt_lang], src_text=src)

        last_error = None  # 用于存储最后一次尝试的错误

        for sleep_time in sleep_times:
            try:
                if "qwen3" in model_name:
                    reasoning_content, outputs = call_qwen3_thinking(text)
                else:
                    reasoning_content, outputs = call_r1(text)
                break  # 成功调用时跳出循环
            except Exception as e:
                last_error = e  # 记录最后一次错误
                print(f"Retry after sleeping {sleep_time} sec...")
                if "Error code: 400" in str(e) or "Error code: 429" in str(e):
                    time.sleep(sleep_time)
                else:
                    error_file[i] = str(e)
                    reasoning_content=""
                    outputs = ""
                    break
        else:
            # 如果达到最大重试次数仍然失败，记录空结果, break不会进入else
            print(f"Skipping {i}")
            outputs = ""
            reasoning_content = ""
            if last_error:  # 确保 last_error 不是 None
                error_file[i] = str(last_error)

        result.append({"idx": i, src_lang: src, tgt_lang: tgt, "reasoning": reasoning_content, "mt": outputs})
        i+=1
    return result

def main(model_name, today):
    root = f"/mnt/workspace/xintong/pjh/All_result/mt_grpo/{model_name}无限制指令-{today}/"
    Path(root).mkdir(parents=True, exist_ok=True)
    print("路径保存地址在", root)
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

        # if lang == "zh2en" or lang=="en2zh":
        #     continue
        print(f"Processing file: {data_file}", lang)
        # Parse the file and translate
        save_name = data_file.split("/")[-1].replace(".jsonl", f"_mt.json")


        result = translate(data_file, lang)    
        json.dump(result, open(root + save_name, "w", encoding="utf-8"), ensure_ascii=False, indent=4)


if __name__ == '__main__':

    error_file = {}

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--terminal', 
        type=int, 
        required=True,  # 如果一定要提供terminal参数
        choices=list(range(1, 7)),  # 限定可选值为 1~6
        help="Specify which terminal block (1 to 6) to run"
    )
        
    # 解析命令行参数
    args = parser.parse_args()
    terminal = args.terminal

    today=datetime.date.today()
    if terminal == 1:
        model_name = "deepseek-r1"
        print(model_name)
        main(model_name, today)
    elif terminal == 2:
        model_name = "qwen3-235b-a22b"
        print(model_name)
        main(model_name, today)
    else:
        print("choose 1/2")
