#!/bin/bash

set -x

model_name=newest_model_name_here  # Replace with your actual model name
# model_name=/mnt/data/users/liamding/data/models/Qwen2.5-7B-Instruct
CUDA_VISIBLE_DEVICES=0 vllm serve ${model_name} --dtype bfloat16 --gpu-memory-utilization 0.9 --tensor-parallel-size 1