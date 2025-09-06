#!/bin/bash

echo "🧪 测试GPU选择和模型加载..."

# 测试GPU选择脚本
echo "1. 测试GPU选择脚本..."
./select_gpus.sh

echo ""
echo "2. 测试环境变量..."
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "SELECTED_GPUS: $SELECTED_GPUS"

echo ""
echo "3. 测试Python环境..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device count: {torch.cuda.device_count()}')
    print(f'Current device: {torch.cuda.current_device()}')
    print(f'Device name: {torch.cuda.get_device_name()}')
    print(f'CUDA_VISIBLE_DEVICES: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"N/A\"}')
"

echo ""
echo "4. 测试模型加载..."
python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B', trust_remote_code=True)
print('✓ Tokenizer loaded successfully')

print('Loading model...')
model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-3B',
    torch_dtype=torch.bfloat16,
    attn_implementation='flash_attention_2',
    device_map='auto',
    trust_remote_code=True
)
print('✓ Model loaded successfully')

print(f'Model device: {next(model.parameters()).device}')
print(f'Model dtype: {next(model.parameters()).dtype}')
"

echo ""
echo "✅ 测试完成！"


