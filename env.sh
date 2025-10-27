#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="xjf_verl_qe"
PY_VER="3.10"
TORCH_VER="2.6.0"
CUDA_TAG="cu124"   # 如果新机不是 CUDA 12.4，请改成 cu121/cu118 等
VLLM_VER="0.8.4"

echo "==> Create/Update conda env: ${ENV_NAME}"
conda env remove -n "${ENV_NAME}" -y >/dev/null 2>&1 || true
conda env create -f environment.yml

echo "==> Activate env"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "==> Install PyTorch ${TORCH_VER} (${CUDA_TAG})"
pip install --upgrade pip
pip install \
  torch==${TORCH_VER} torchvision==0.21.0 torchaudio==${TORCH_VER} \
  --index-url https://download.pytorch.org/whl/${CUDA_TAG}

echo "==> Install project Python deps"
pip install -r requirements.lock.txt

# # 本地/私有轮子（如果有）
# if compgen -G "wheels/*.whl" > /dev/null; then
#   echo "==> Install local wheels"
#   pip install wheels/*.whl
# fi

echo "==> Quick check"
python - <<'PY'
import torch, transformers, vllm, numpy as np
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda, "is_available:", torch.cuda.is_available())
print("Transformers:", transformers.__version__)
print("vLLM:", vllm.__version__)
PY

# INFO 10-27 10:36:07 [__init__.py:239] Automatically detected platform cuda.
# Torch: 2.6.0+cu124 CUDA: 12.4 is_available: False
# Transformers: 4.51.1
# vLLM: 0.8.4
echo "All set ✅"