# from comet_reward_batch_with_ray_debug import compute_score

# # 训练模式（返回 List[float]）
# print(compute_score(
#     data_sources=["test_en-zh", "test_zh-en"],   # 只是标签，用于分桶统计
#     solution_strs=[
#         "<think>t</think><translate>Hello</translate>",
#         "<think>t</think><translate>你好</translate>",
#     ],
#     ground_truths=["Hallo", "Hello"],            # 参考译文
#     extra_infos=[
#         {"source": "Hello", "lg": "en-zh"},      # 这里才是“原句”
#         {"source": "你好",  "lg": "zh-en"},
#     ],
#     compute_val_reward=False,
# ))

# # 验证模式（需要各指标明细）
# out = compute_score(
#     data_sources=["test_en-zh"],
#     solution_strs=["<think>t</think><translate>World</translate>"],
#     ground_truths=["Welt"],
#     extra_infos=[{"source": "World", "lg": "en-zh"}],
#     compute_val_reward=True,
#     return_dict=True,
# )
# print(out.keys())  # dict_keys(['reward_tensor', 'reward_extra_info', 'scores'])
# print(out["reward_extra_info"])  # {'score': [...], 'format_score': [...], 'bleu_score': [...], 'comet_score': [...], 'word_level_qe': [...]}
try:
    import flashinfer
    print("✅ flashinfer loaded:", flashinfer.__version__)
except ImportError:
    print("❌ flashinfer not found")