CUDA_VISIBLE_DEVICES=0 python3 -m verl.model_merger merge \
--backend fsdp \
--local_dir /mnt/workspace/xintong/pjh/All_result/mt_grpo/verl_grpo_xwang/qwen2.5_7b_r1-zero/global_step_102/actor \
--target_dir /mnt/workspace/xintong/pjh/All_result/mt_grpo/verl_grpo_xwang/merge_model/qwen2.5_7b_r1-zero