from comet.models import load_from_checkpoint
import torch, os
from itertools import chain

ckpt = os.path.expanduser('~/MT_Grpo_qe/ckpts/comet/WMT24-QE-task2-baseline/checkpoints/model.fixed.ckpt')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = load_from_checkpoint(ckpt).to(device)
model.eval()

data = [
    {"src": "The output signal provides constant sync so the display never glitches.",
     "mt":  "Das Ausgangssignal bietet eine konstante Synchronisation, so dass die Anzeige nie stört."},
    {"src": "Kroužek ilustrace je určen všem milovníkům umění ve věku od 10 do 15 let.",
     "mt":  "Кільце ілюстрації призначене для всіх любителів мистецтва у віці від 10 до 15 років."},
    {"src": "Mandela then became South Africa's first black president after his African National Congress party won the 1994 election.",
     "mt": "その後、1994年の選挙でアフリカ国民会議派が勝利し、南アフリカ初の黒人大統領となった。"
    }
]

out = model.predict(data, batch_size=8, gpus=1)

print(type(out), out.keys())             # dict_keys(['score', 'tags', 'system_score'])
print("num items:", len(out["score"]))   # = len(data)
for i in range(len(out["score"])):
    print("Item:", i)
    print("score:", out["score"][i])   # float
    print("tags len:", len(out["tags"][i]))    # 词/子词数
    print("tags:", out["tags"][i])
    print("system_score:", out["system_score"])     # ['OK','BAD',...]
    
'''
输出结果：
num items: 3
Item: 0
score: 0.30034542083740234
tags len: 12
tags: ['OK', 'OK', 'OK', 'OK', 'OK', 'OK', 'OK', 'OK', 'OK', 'BAD', 'BAD', 'BAD']
system_score: 0.6509629885355631
Item: 1
score: 0.8767302632331848
tags len: 14
tags: ['OK', 'OK', 'OK', 'OK', 'OK', 'OK', 'OK', 'OK', 'OK', 'OK', 'OK', 'OK', 'OK', 'OK']
system_score: 0.6509629885355631
Item: 2
score: 0.7758132815361023
tags len: 1 # <--可以看到没有空格的日文句子直接被划分为了一个词
tags: ['OK']
system_score: 0.6509629885355631
'''
# model.predict(...) 的返回值其实是一个字典，键包括：
# 	•	score：句级分数， QE 预测值（拟合 DA/MQM/HTER 的回归输出
# 	•	tags：词级 OK/BAD 标签（通常是 List[List[str]]，外层按样本，内层按词/子词）
# 	•	system_score：样本内句级 score 的平均值（方便当系统级分）。


# import jieba
# import re
# import fugashi
# tagger = fugashi.Tagger()

# # ---------- 判断文本主语言 ----------
# def detect_primary_lang(text: str) -> str:
#     """
#     整段文本主语言：
#       - 'ja': 若含假名 -> 日文（即使也有汉字）
#       - 'zh': 否则若含汉字 -> 中文
#       - 'latin': 否则若含拉丁字母 -> 英/德
#       - 'other': 其他
#     """
#     # 先检查假名（日本特有）
#     if re.search(r"[\p{Hiragana}\p{Katakana}]", text):
#         return "ja"
#     # 再检查汉字
#     if re.search(r"[\p{Han}]", text):
#         return "zh"
#     # 再检查拉丁字母
#     if re.search(r"[A-Za-z\u00C0-\u024F]", text):
#         return "latin"
#     return "other"

# # ---------- 语言内分词 ----------
# def whitespace_segments(text: str) -> List[str]:
#     """按空白切分成块（仅作为外层块划分，不做词法判定）"""
#     return text.split(" ")
    
# def zh_segments(text: str) -> List[str]:
#     # 中文分词（jieba）
#     return list(jieba.cut(text))

# def ja_segments(text: str, _tagger_cache: dict = {}) -> List[Tuple[int, int]]:
#     # 日文分词（fugashi）
#     return [word.surface for word in tagger(text)]

