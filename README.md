# MT_Grpo

## 2025.6.24
1. `conda env create -f environment.yml` 创建的环境名字叫`pjh_grpo_mt`
2. 下载模型 `Unbabel/wmt23-cometkiwi-da-xl`
3. 运行8卡训练 `bash grpo.sh` ，生成地址在`/mnt/workspace/xintong/pjh/All_result/mt_grpo/grpo_output/qwen2.5-7b-inst/`
4. 手动合并，修改`merge.sh`里面的`--adapters`后面的地址为权重地址，然后运行`CUDA_VISIBLE_DEVICES=0 bash merge.sh`
5. 输出的合并地址为：`/mnt/workspace/xintong/pjh/All_result/mt_grpo/grpo_output/merged_lora/qwen2.5-7b-inst/`
