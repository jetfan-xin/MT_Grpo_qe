
set -x
# 激活 Conda 环境
echo "🔄 正在切换到 Conda 环境 pjh_verl..."
eval "$(conda shell.bash hook)"
conda activate pjh_verl

# 检查 conda 环境是否激活成功
if [[ "$CONDA_DEFAULT_ENV" == "pjh_verl" ]]; then
  echo "✅ Conda 环境 pjh_verl 已成功激活！"
else
  echo "❌ Conda 环境激活失败！当前环境为：$CONDA_DEFAULT_ENV"
  exit 1
fi

comet_rm=False
comet_free_rm=True 
train_file_path=../data/train/parquet/train_base_enzh_zhen.parquet
test_file_path=../data/test/parquet/test_base_enzh_zhen.parquet
python3 ../data/process_data.py \
    --train_files "../data/train/json/train_zhen_6565.jsonl" "../data/train/json/train_enzh_6565.jsonl" \
    --test_files "../data/test/json/wmt23_zhen.jsonl" "../data/test/json/wmt24_enzh.jsonl" \
    --tokenizer_path Qwen/Qwen2.5-7B \
    --template_type "base" \
    --train_output_file ${train_file_path} \
    --test_output_file ${test_file_path}

export WANDB_API_KEY=1526cd13c8d1f8c8529ea57f23d553b20b03451c # set your wandb api key
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=../data/train/parquet/train_base_enzh_zhen.parquet \
    data.val_files=../data/test/parquet/test_base_enzh_zhen.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=128  \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=128 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=128 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    custom_reward_function.path=comet_reward_batch.py \
    reward_model.reward_manager=batch \
    trainer.val_before_train=False \
    trainer.logger=['wandb'] \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=40 \
    trainer.test_freq=10 \
    trainer.val_before_train=False \
    trainer.default_local_dir=/mnt/workspace/xintong/pjh/All_result/mt_grpo/verl_grpo_xwang/qwen2.5_7b_r1-zero \
    trainer.total_epochs=1 $@ 2>&1 | tee custom_grpo_fast.log
  

# trainer.project_name='verl_grpo_xwang' \
# trainer.experiment_name='qwen2.5_7b_r1-zero' \