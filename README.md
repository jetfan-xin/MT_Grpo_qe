# MT_Grpo
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
