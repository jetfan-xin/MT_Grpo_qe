#!/usr/bin/env bash
set -euxo pipefail

########## 0) 环境与项目基础 ##########
# 选GPU：把列表改成你要用的显卡，比如 "0,1" 或 "0,1,2,3"
export JB_LIGHTWEIGHT=0    # 关闭轻量模式
export CUDA_VISIBLE_DEVICES=1,2,4
unset RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=1          # 单机、无 IB 就关掉
export NCCL_P2P_DISABLE=1         # 先关 P2P，避免拓扑坑；跑通后可再尝试打开
export NCCL_SHM_DISABLE=1         # 有些环境共享内存也会坑
export TORCH_NCCL_BLOCKING_WAIT=1       # 出问题尽快报错而不是无限等
export CUDA_DEVICE_MAX_CONNECTIONS=1

# # Ray 默认会改 CUDA_VISIBLE_DEVICES；加这一句让 Ray 不改你手动选择
# export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
python - <<'PY'
import torch; print("torch cuda count =", torch.cuda.device_count())
PY
# wandb：换成你自己的 key / 项目名 / entity
export WANDB_API_KEY=a28f5c63c96f3bdc885978f31f4619b48811cff7
export WANDB_PROJECT=qwen25_gapo_mt
export WANDB_ENTITY=jetfan-universit-t-hamburg
export WANDB_MODE=online
# 可选：指定 wandb 的本地缓存目录
export WANDB_DIR="$PWD/wandb_cache"

# （原来脚本里的 swanlab 就别用了）
# export SWANLAB_API_KEY=57bftOCtg6exWFs81mtT1

########## 1) 预处理数据 ##########
train_file_path=../data/train/parquet/train_base_enzh_zhen.parquet
test_file_path=../data/test/parquet/test_base_enzh_zhen.parquet
python3 ../data/process_data.py \
  --train_files "../data/train/json/train_zhen_6565.jsonl" "../data/train/json/train_enzh_6565.jsonl" \
  --test_files "../data/test/json/wmt23_zhen.jsonl" "../data/test/json/wmt24_enzh.jsonl" \
  --tokenizer_path Qwen/Qwen2.5-0.5B \
  --template_type "base" \
  --train_output_file ${train_file_path} \
  --test_output_file ${test_file_path}

########## 2) 训练超参 ##########
adv_estimator=grpo

# 高熵 token 过滤（RLVR）
enable_entropy_mask=true
top_entropy_quantile=0.2

use_kl_in_reward=false
kl_coef=0.0
use_kl_loss=false
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 8)) #((1024 * 20))
enable_overlong_buffer=true
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

enable_filter_groups=true
filter_groups_metric=seq_reward #acc
max_num_gen_batches=4 #10
train_prompt_bsz=32 # 128
gen_prompt_bsz=$((train_prompt_bsz * 3))
n_resp_per_prompt=12 #16
train_prompt_mini_bsz=16 #32

# Ray / 分布式
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/recipe/dapo/runtime_env.yaml"}
NNODES=${NNODES:-1}

# 路径：明确使用 Qwen2.5-3B
MODEL_PATH="Qwen/Qwen2.5-0.5B"
TRAIN_FILE="${train_file_path}"
TEST_FILE="${test_file_path}"

# 采样
temperature=1.0
top_p=1.0
top_k=-1        # 注意：下面传参不要再加引号，保持整数

# 性能
# sp_size = 每节点使用的 GPU 数，要与上面的 CUDA_VISIBLE_DEVICES 数量一致
sp_size=2 #4
use_dynamic_bsz=true
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
offload=true
gen_tp=1

sp_actor=2
sp_ref=2

python - <<'PY'
import os, ray
ray.init(ignore_reinit_error=True)
print("cluster GPUs:", ray.cluster_resources().get("GPU"))
@ray.remote(num_gpus=1)
def see():
    import os, torch
    print("PID",os.getpid(),"CVD=",os.getenv("CUDA_VISIBLE_DEVICES"),
          "count=", torch.cuda.device_count())
    return os.getenv("CUDA_VISIBLE_DEVICES")
print(ray.get([see.remote() for _ in range(4)]))
PY

########## 3) 启动训练（wandb + Qwen2.5‑3B + FA2 + FP16 + CUDA） ##########
# -   trainer.default_local_dir=./qwen2.5_0.5b_dapo_bleu_comet \代表模型存储目录。
# -   存储剩余的30G大概也不够存储qwen-3b的一个checkpoint，遂修改trainer.save_freq=20为较大的数字，确认模型能跑通即可。
python3 -m recipe.dapo.main_dapo \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${TEST_FILE}" \
  data.truncation='left' \
  data.max_prompt_length=${max_prompt_length} \
  data.max_response_length=${max_response_length} \
  data.gen_batch_size=${gen_prompt_bsz} \
  data.train_batch_size=${train_prompt_bsz} \
  actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
  algorithm.adv_estimator=${adv_estimator} \
  algorithm.enable_entropy_mask=${enable_entropy_mask} \
  algorithm.top_entropy_quantile=${top_entropy_quantile} \
  algorithm.use_kl_in_reward=${use_kl_in_reward} \
  algorithm.kl_ctrl.kl_coef=${kl_coef} \
  actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
  actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
  actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
  actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
  actor_rollout_ref.actor.clip_ratio_c=10.0 \
  algorithm.filter_groups.enable=${enable_filter_groups} \
  algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
  algorithm.filter_groups.metric=${filter_groups_metric} \
  actor_rollout_ref.model.use_remove_padding=true \
  actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps=2 \
  actor_rollout_ref.actor.optim.weight_decay=0.1 \
  actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
  actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.grad_clip=1.0 \
  actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_actor} \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
  actor_rollout_ref.rollout.enable_chunked_prefill=true \
  actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length+max_response_length)) \
  actor_rollout_ref.rollout.temperature=${temperature} \
  actor_rollout_ref.rollout.top_p=${top_p} \
  actor_rollout_ref.rollout.top_k=${top_k} \
  actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
  actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
  actor_rollout_ref.rollout.val_kwargs.do_sample=true \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
  actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_ref} \
  actor_rollout_ref.actor.fsdp_config.fsdp_size=3 \
  custom_reward_function.path=comet_reward_batch.py \
  reward_model.reward_manager=dapo \
  reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
  reward_model.overlong_buffer.len=${overlong_buffer_len} \
  reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
  trainer.logger=['wandb'] \
  trainer.project_name=${WANDB_PROJECT} \
  trainer.experiment_name="qwen25_0.5b_gapo_highent" \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes="${NNODES}" \
  trainer.val_before_train=true \
  trainer.test_freq=5 \
  trainer.save_freq=100 \
  trainer.total_epochs=1 \
  trainer.default_local_dir=./qwen2.5_0.5b_dapo_bleu_comet \
  trainer.validation_data_dir=./qwen2.5_0.5b_dapo_bleu_comet/validation_samples \
  trainer.log_val_generations=100 \
  trainer.resume_mode=auto \
  actor_rollout_ref.rollout.dtype=float16 \
  "$@" 2>&1 | tee custom_dapo_fast_rlvr_wandb.log
