#!/bin/bash

# 激活 Conda 环境
echo "🔄 正在切换到 Conda 环境 pjh_grpo_mt..."
eval "$(conda shell.bash hook)"
conda activate pjh_grpo_mt

# 检查 conda 环境是否激活成功
if [[ "$CONDA_DEFAULT_ENV" == "pjh_grpo_mt" ]]; then
  echo "✅ Conda 环境 pjh_grpo_mt 已成功激活！"
else
  echo "❌ Conda 环境激活失败！当前环境为：$CONDA_DEFAULT_ENV"
  exit 1
fi

model_name=/mnt/workspace/xintong/pjh/All_result/mt_grpo/grpo_output/qwen2.5-7b-inst/v2-20250627-102443/checkpoint-3249
CUDA_VISIBLE_DEVICES=0,1 vllm serve ${model_name} --dtype bfloat16 --gpu-memory-utilization 0.9 --tensor-parallel-size 2