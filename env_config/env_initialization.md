### 环境安装步骤

```bash
# 1) 用 environment.yml 创建环境
conda env create -f environment.yml

# 如果中断，继续如下：

# 1) 进入环境
conda activate verl-qe

# 2) 先装 PyTorch（和原环境一致：CUDA 12.4 + torch 2.6.0）
pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

# 3) 把 yml 里剩余的 pip 包继续安装
pip install -r pip_full_clean.txt   # 可以清除掉已安装的包
```