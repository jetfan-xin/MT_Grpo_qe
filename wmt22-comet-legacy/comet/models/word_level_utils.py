import torch
from more_itertools import collapse
import numpy as np
import math
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
    
# def zh_segments(text: str) -> List[Tuple[int, int]]:
#     """中文：jieba tokenize；无 jieba 时退化为逐字符（不丢标点/数字/字母）"""
#     return list(jieba.cut(text))
    

# def ja_segments(text: str, _tagger_cache: dict = {}) -> List[Tuple[int, int]]:
#     """日文：fugashi(MeCab)。无论是否有 .offset，都用逐字节累加得到 (start, end)。失败则退化为逐字符"""
#     # 尽量保持与原逻辑一致：若 fugashi 不可用/初始化失败 -> 逐字符
#     try:
#         from fugashi import Tagger as FugashiTagger
#         tagger = _tagger_cache.get("tagger")
#         if tagger is None:
#             # 如果你是 brew 装的 mecab，且 mecabrc 路径特殊，可改：FugashiTagger("-r /opt/homebrew/etc/mecabrc")
#             tagger = FugashiTagger()
#             _tagger_cache["tagger"] = tagger
#     except Exception:
#         return [(i, i + 1) for i in range(len(text)) if not text[i].isspace()]

#     spans: List[Tuple[int, int]] = []
#     cur = 0
#     try:
#         for node in tagger(text):
#             surf = str(node.surface)
#             length = len(surf)
#             s = cur
#             e = s + length
#             spans.append((s, e))
#             cur = e
#         return spans
#     except Exception:
#         # 任意异常都退回逐字符
#         return [(i, i + 1) for i in range(len(text)) if not text[i].isspace()]


def convert_word_tags(wt_list):
        word_tags = []
        d = {'BAD': 1, 'OK': 0}
        for l in wt_list:
            word_tags.append([d[x] for x in l.split(' ')]) ####替换这里就好啦！
        return word_tags


def flatten(self, list_of_lists):
    for list in list_of_lists:
        for item in list:
            yield item

def confusion_matrix(hat_y_all, y_all, n_classes=None):
    cnfm = np.zeros((n_classes, n_classes))

    for hat_y, y in zip(hat_y_all, y_all):
        hat_y = hat_y.view(-1,2).cpu()
        y = y.view(-1).cpu()
        for j in range(y.size(0)):
            if y[j]>-1:
                hat_yj = np.argmax(hat_y[j])
                cnfm[int(y[j]), hat_yj] += 1
    return cnfm

def matthews_correlation_coefficient(hat_y, y):
    """Compute Matthews Correlation Coefficient.

    Arguments:
        hat_y: list of np array of predicted binary labels.
        y: list of np array of true binary labels.

    Return:
        the Matthews correlation coefficient of hat_y and y.
    """

    cnfm = confusion_matrix(hat_y, y, 2)
    tp = cnfm[0][0]
    tn = cnfm[1][1]
    fp = cnfm[1][0]
    fn = cnfm[0][1]
    class_p = tp + fn
    class_n = tn + fp
    pred_p = tp + fp
    pred_n = tn + fn
    
    normalizer = class_p * class_n * pred_p * pred_n
  
    if normalizer:
        return ((tp * tn) - (fp * fn)) / math.sqrt(normalizer)
    else:
        return 0
