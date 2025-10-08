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
    {"src": "曼德拉随后在 1994 年非洲人国民大会党赢得选举后，成为南非首位黑人总统。",
     "mt":  "Mandela then became South Africa's first black president after his African National Congress party won the 1994 election."},
    {"src": "Mandela then became South Africa's first black president after his African National Congress party won the 1994 election.",
     "mt": "曼德拉随后在 1994 年非洲人国民大会党赢得选举后，成为南非首位黑人总统。"
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
score: 0.3027360439300537
tags len: 12
tags: ['OK', 'OK', 'OK', 'OK', 'OK', 'OK', 'OK', 'OK', 'OK', 'BAD', 'BAD', 'BAD']
system_score: 0.6735697587331136

Item: 1
score: 0.8658638000488281
tags len: 18
tags: ['OK', 'OK', 'OK', 'OK', 'OK', 'OK', 'OK', 'OK', 'OK', 'OK', 'OK', 'OK', 'OK', 'OK', 'OK', 'OK', 'OK', 'OK']
system_score: 0.6735697587331136

Item: 2
score: 0.852109432220459
tags len: 3
tags: ['BAD', 'OK', 'OK'] <- 按空格划分中/日文句子为spans
system_score: 0.6735697587331136
'''
# model.predict(...) 的返回值其实是一个字典，键包括：
# 	•	score：句级分数， QE 预测值（拟合 DA/MQM/HTER 的回归输出)
# 	•	tags：词级 OK/BAD 标签（通常是 List[List[str]]，外层按样本，内层按词/子词）
# 	•	system_score：样本内句级 score 的平均值（方便当系统级分）。
