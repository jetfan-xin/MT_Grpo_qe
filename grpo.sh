# WANDB_MODE=offline \
    # --sleep_level 1 \
    # --offload_optimizer true \
    # --offload_model true \
    # --gc_collect_after_offload true \

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

WANDB_DIR=/mnt/workspace/xintong/pjh/All_result/mt_grpo/wandb \
WANDB_API_KEY=xxx \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  \
NPROC_PER_NODE=8 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset data/mt-r1-zero-train.jsonl \
    --external_plugins ms-swift/examples/train/grpo/plugin/plugin.py \
    --reward_funcs comet \
    --reward_weights 1 \
    --train_type full \
    --torch_dtype bfloat16 \
    --use_vllm true \
    --vllm_gpu_memory_utilization 0.5 \
    --log_completions true \
    --max_completion_length 1024 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-7 \
    --eval_steps 500 \
    --save_steps 1000 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --output_dir /mnt/workspace/xintong/pjh/All_result/mt_grpo/grpo_output/qwen2.5-7b-inst/ \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 8 \
    --temperature 1.0 \
    --top_p 0.9 \
    --top_k 50 \
    --deepspeed zero3 \
    --report_to wandb

