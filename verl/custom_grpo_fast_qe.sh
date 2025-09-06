#!/usr/bin/env bash
set -euxo pipefail
# #!/bin/bash
# set -x

########## 0) 环境与项目基础 ##########
# export HF_HOME=/tmp/hf_home_$USER
# export HF_DATASETS_CACHE=$HF_HOME/datasets
# export TRANSFORMERS_CACHE=$HF_HOME/transformers
# export VLLM_CACHE_DIR=$HF_HOME/vllm
# export RAY_TMPDIR=/tmp/ray_$USER
# mkdir -p "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$VLLM_CACHE_DIR" "$RAY_TMPDIR"
# df -h /ltstorage/home/4xin /tmp

export JB_LIGHTWEIGHT=0    # 关闭轻量模式
export CUDA_VISIBLE_DEVICES=1,2,3,4 # NCCL 设置（推荐保留）
RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export NCCL_DEBUG=WARN # INFO
export NCCL_DEBUG_SUBSYS=GRAPH,COLL
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# 尽量先别禁用 NCCL 能力（之前这些容易把通信逼到慢路径，导致更容易卡在捕获点）
# unset NCCL_P2P_DISABLE
unset NCCL_SHM_DISABLE
# unset NCCL_IB_DISABLE
# 避免复杂直连/跨架构 peer 访问引发的奇怪等待读写
export NCCL_P2P_DISABLE=1
# 或者完全走 SHM（本机）禁 IB（如果没有 IB，就禁用）
export NCCL_IB_DISABLE=1

export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1


# wandb：换成你自己的 key / 项目名 / entity
export WANDB_API_KEY=a28f5c63c96f3bdc885978f31f4619b48811cff7
export WANDB_PROJECT="qwen2.5_0.5b_grpo_mt"
export WANDB_NAME="qwen2.5_0.5b_r1-zero"
export WANDB_ENTITY="jetfan-universit-t-hamburg"
export WANDB_MODE=online
# 可选：指定 wandb 的本地缓存目录
export WANDB_DIR="$PWD/wandb_cache"


# 数据路径
comet_rm=False
comet_free_rm=True 
gpu_count=4

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


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${train_file_path} \
    data.val_files=${test_file_path} \
    data.train_batch_size=16 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size=4  \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=4\
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    custom_reward_function.path=comet_reward_batch.py \
    reward_model.reward_manager=batch \
    trainer.val_before_train=False \
    trainer.logger=['wandb'] \
    trainer.project_name=${WANDB_PROJECT} \
    trainer.experiment_name=${WANDB_NAME} \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=10 \
    trainer.val_before_train=False \
    trainer.default_local_dir=./results/qwen2.5_0.5b_r1-zero \
    trainer.total_epochs=1 $@ 2>&1 | tee custom_grpo_fast_qe.log