# MT_Grpo
## 2025.8.15
训练GTPO-Qwen3b
1. `conda activate pjh_verl`
2. `cd verl`，然后`bash custom_gtpo_fast.sh`，训练权重保存位置在`/mnt/workspace/xintong/pjh/All_result/mt_grpo/verl_grpo_xwang/qwen2.5_3b_gtpo_bleu_comet_entropy_b1`
3. 训练成功之后，还是在`verl`文件夹下面，合并模型权重，`bash merge.sh`，保存模型在`/mnt/workspace/xintong/pjh/All_result/mt_grpo/verl_grpo_xwang/merge_model/qwen2.5_3b_gtpo_bleu_comet_entropy_b1`
4. 测试模型生成，在项目的根目录下（`MT_Grpo`）运行`CUDA_VISIBLE_DEVICES=0 python vllm_infer.py`，保存的结果在`/mnt/workspace/xintong/pjh/All_result/mt_grpo/verl_grpo_result/qwen2.5_3b_gtpo_bleu_comet_entropy_b1/`

## 2025.8.3
测试训练效果
1. `conda activate pjh_verl`
2. `cd verl`，然后`bash merge.sh`，合并模型权重，保存位置在`/mnt/workspace/xintong/pjh/All_result/mt_grpo/verl_grpo_xwang/merge_model/qwen2.5_7b_r1-zero`
3. 运行 `CUDA_VISIBLE_DEVICES=0 python qwen25_gen.py`，保存的文件夹是`/mnt/workspace/xintong/pjh/All_result/mt_grpo/verl_grpo_result/qwen2.5_7b_r1-zero_verl/`


## 2025.7.24
解决comet没有用GPU，并且进行batch运算
- 在verl目录下 `bash custom_grpo_fast.sh`
- 保存的路径应该是 `/mnt/workspace/xintong/pjh/All_result/mt_grpo/verl_grpo_xwang/qwen2.5_7b_r1-zero`
- 生成`custom_grpo_fast.log` 实时查看终端输出


## 2025.7.11
增大推理时vllm的占用率，增大更新actor时的batch, 但是不确定是否会OOM
`bash custom_grpo3.sh`

## 2025.7.5
增大batch size 训练verl, 但是不确定是否会OOM
`bash custom_grpo2.sh`

## 2025.7.5
训练verl
1. 安装
```bash
conda create -n pjh_verl python==3.10 -y
cd verl
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .
pip install sacrebleu unbabel-comet
```
2. 训练
在verl目录下 `bash custom_grpo.sh`


## 2025.6.28
测试grpo模型的效果
1. 在`start_vllm.sh`文件里面指定训练好的模型地址到model_name：`/mnt/workspace/xintong/pjh/All_result/mt_grpo/grpo_output/qwen2.5-7b-inst/{version-时间戳}/{checkpoint最新的} `
2. 在一个terminal运行`bash start_vllm.sh`（默认用了0，1号gpu）
3. 等第二步服务起来之后，开一个terminal分别运行`python translate.py --lang zh2en`和`python translate.py --lang en2zh`。结果保存在`/mnt/workspace/xintong/pjh/All_result/mt_grpo/eval_qwen2.5-7b_grpo/`

## 2025.6.24
1. `conda env create -f environment.yml` 创建的环境名字叫`pjh_grpo_mt`
2. 下载模型 `Unbabel/wmt23-cometkiwi-da-xl`
3. 运行8卡训练 `bash grpo.sh` ，生成地址在`/mnt/workspace/xintong/pjh/All_result/mt_grpo/grpo_output/qwen2.5-7b-inst/`
