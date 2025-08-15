CUDA_VISIBLE_DEVICES=0 python3 -m verl.model_merger merge \
--backend fsdp \
--local_dir /mnt/workspace/xintong/pjh/All_result/mt_grpo/verl_grpo_xwang/qwen2.5_3b_gtpo_bleu_comet_entropy_b1/global_step_34/actor \
--target_dir /mnt/workspace/xintong/pjh/All_result/mt_grpo/verl_grpo_xwang/merge_model/qwen2.5_3b_gtpo_bleu_comet_entropy_b1