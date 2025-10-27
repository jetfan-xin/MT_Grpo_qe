# from comet import download_model, load_from_checkpoint
from comet import download_model, load_from_checkpoint

model_path = download_model("Unbabel/XCOMET-XL")
model = load_from_checkpoint(model_path)
data = [
    {
        "src": "Boris Johnson teeters on edge of favour with Tory MPs", 
        "mt": "Boris Johnson ist bei Tory-Abgeordneten völlig in der Gunst", 
        "ref": "Boris Johnsons Beliebtheit bei Tory-MPs steht auf der Kippe"
    }
]
model_output = model.predict(data, batch_size=8, gpus=1)
# Segment-level scores
print (model_output.scores)

# System-level score
print (model_output.system_score)

# Score explanation (error spans)
print (model_output.metadata.error_spans)

# model_path = "/mnt/data1/users/4xin/hf/hub/models--Unbabel--XCOMET-XXL/snapshots/873bac1b1c461e410c4a6e379f6790d3d1c7c214/checkpoints/model.ckpt"
# model_path = download_model("Unbabel/XCOMET-XL")
# print("model_dir =", model_path)
# model = load_from_checkpoint(model_path)
# data = [
#     {"src": "The output signal provides constant sync so the display never glitches.",
#      "mt":  "Das Ausgangssignal bietet eine konstante Synchronisation, so dass die Anzeige nie stört."},
#     {"src": "曼德拉随后在 1994 年非洲人国民大会党赢得选举后，成为南非首位黑人总统。",
#      "mt":  "Mandela then became South Africa's first black president after his African National Congress party won the 1994 election."},
#     {"src": "Mandela then became South Africa's first black president after his African National Congress party won the 1994 election.",
#      "mt": "曼德拉随后在 1994 年非洲人国民大会党赢得选举后，成为南非首位黑人总统。"
#     },
#     {
#     "src": "The Great Wall of China was built over several centuries to protect the Chinese states from invasions.",
#     "mt": "中国的长城历经数个世纪建成，用以保护各个中国王国免受外敌入侵。"
#     },
#     {
#     "src": "it seems like even iMessage over WiFi isn't working, which doesn't quite make sense to me",
#     "mt": "连接WiFi的iMessage好像都不能用，这没道理啊"
#     },
#     {
#     "src": "The international financial institution is poised to launch a strategy aimed at significantly advancing the rights and economic opportunities for girls and women by 2030. The plan focuses on improving inclusivity and resilience through access to finance, technological innovation, and changes in public policy and law.",
#     "mt": "世界银行准备推出一项战略，旨在到2030年大幅提升女童和妇女的权利和经济机会。该计划的重点是通过获得资金、技术创新以及公共政策和法律修订，提高包容性和复原力。"
#     },
#     {
#     "src": "It’s been raining all day. I just want to stay home and watch movies.",
#     "mt": "一整天都在下雨，我只想待在家里看电影。"
#     }
# ]
# model_output = model.predict(data, batch_size=8, gpus=1)
# # Segment-level scores
# print (model_output.scores)
# # System-level score：样本内句级 score 的平均值（方便当系统级分）。
# print (model_output.system_score)
# # Score explanation (error spans)
# print (model_output.metadata.error_spans)


# '''
# 输出结果：
# [0.9782446622848511, 1.0, 0.7227798104286194, 1.0, 0.9271236658096313, 0.7638593912124634, 1.0]
# 0.9131439328193665
# [
#  [{'text': 'stört', 'confidence': 0.9638608694076538, 'severity': 'minor', 'start': 81, 'end': 87}], 
#  [], 
#  [{'text': '随后在', 'confidence': 0.4205929636955261, 'severity': 'major', 'start': 3, 'end': 6}, 
#   {'text': '年非洲人', 'confidence': 0.3568190932273865, 'severity': 'major', 'start': 12, 'end': 16}, 
#   {'text': '大会党', 'confidence': 0.3804951310157776, 'severity': 'major', 'start': 18, 'end': 21}]
#  [], 
#  [{'text': '连接WiFi的', 'confidence': 0.4864675998687744, 'severity': 'minor', 'start': 0, 'end': 7}, 
#   {'text': '这没道理啊', 'confidence': 0.38914361596107483, 'severity': 'minor', 'start': 22, 'end': 27}], 
#  [{'text': '世界银行', 'confidence': 0.47887080907821655, 'severity': 'minor', 'start': 0, 'end': 4}, 
#   {'text': '一项', 'confidence': 0.513891339302063, 'severity': 'minor', 'start': 8, 'end': 10}, 
#   {'text': '女童和', 'confidence': 0.6765548586845398, 'severity': 'minor', 'start': 25, 'end': 28}, 
#   {'text': '是通过获得资金', 'confidence': 0.47255000472068787, 'severity': 'major', 'start': 45, 'end': 52}, 
#   {'text': '修订', 'confidence': 0.5757784247398376, 'severity': 'minor', 'start': 66, 'end': 68}, 
#   {'text': '提高包容性和复原力', 'confidence': 0.5188518762588501, 'severity': 'major', 'start': 69, 'end': 78}], 
#  []
# ]
# '''

# import os
# from comet import download_model, load_from_checkpoint
# model_path = os.getenv(
#     "COMET_CKPT",
#     "/mnt/data1/users/4xin/hf/hub/"
#     "models--Unbabel--wmt23-cometkiwi-da-xl/"
#     "snapshots/33858b2239a139d497d9c74952c88b89a8c06213/"
#     "checkpoints/model.ckpt",
# )
# model = load_from_checkpoint(model_path)
# data = [
#     {
#         "src": "The output signal provides constant sync so the display never glitches.",
#         "mt": "Das Ausgangssignal bietet eine konstante Synchronisation, so dass die Anzeige nie stört.",
#         "ref": "nihao niahao nihao."
#     },
#     {
#         "src": "The output signal provides constant sync so the display never glitches.",
#         "mt": "Das Ausgangssignal bietet eine konstante Synchronisation, so dass die Anzeige nie stört."
#     },
#     {
#         "src": "Kroužek ilustrace je určen všem milovníkům umění ve věku od 10 do 15 let.",
#         "mt": "Кільце ілюстрації призначене для всіх любителів мистецтва у віці від 10 до 15 років."
#     },
#     {
#         "src": "Mandela then became South Africa's first black president after his African National Congress party won the 1994 election.",
#         "mt": "その後、1994年の選挙でアフリカ国民会議派が勝利し、南アフリカ初の黒人大統領となった。",
#         "ref": "nihao niahao nihao."
#     }
# ]
# model_output = model.predict(data, batch_size=8, gpus=1)
# print (model_output)