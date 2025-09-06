#!/bin/bash

# 智能GPU选择脚本
# 功能：选择空闲GPU或最合适的GPU组合

echo "🔍 正在分析GPU使用情况..."

# 初始化变量
available_gpus=""
selected_gpus=""
selection_reason=""

# 直接解析gpustat输出
gpustat_output=$(gpustat --no-header)

# 解析每一行GPU信息
while IFS= read -r line; do
    # 跳过空行
    if [ -z "$line" ]; then
        continue
    fi
    
    # 使用正则表达式解析GPU信息
    if [[ $line =~ \[([0-9]+)\].*\|[[:space:]]*([0-9]+)°C,[[:space:]]*([0-9]+)[[:space:]]*%[[:space:]]*\|[[:space:]]*([0-9]+)[[:space:]]*/[[:space:]]*([0-9]+)[[:space:]]*MB ]]; then
        gpu_id="${BASH_REMATCH[1]}"
        temp="${BASH_REMATCH[2]}"
        utilization="${BASH_REMATCH[3]}"
        memory_used="${BASH_REMATCH[4]}"
        memory_total="${BASH_REMATCH[5]}"
        
        # 计算内存使用率
        memory_usage_percent=$((memory_used * 100 / memory_total))
        
        echo "GPU $gpu_id: 温度${temp}°C, 利用率${utilization}%, 内存${memory_used}/${memory_total}MB (${memory_usage_percent}%)"
        
        # 判断是否空闲 (利用率<5% 且 内存使用<10%)
        if [ "$utilization" -lt 5 ] && [ "$memory_usage_percent" -lt 10 ]; then
            if [ -z "$available_gpus" ]; then
                available_gpus="$gpu_id"
            else
                available_gpus="$available_gpus,$gpu_id"
            fi
            echo "  ✅ GPU $gpu_id 空闲"
        else
            echo "  ❌ GPU $gpu_id 被占用"
        fi
    fi
done <<< "$gpustat_output"

echo ""
echo "📊 GPU分析结果:"

# 如果有空闲GPU，选择合适数量的GPU
if [ ! -z "$available_gpus" ]; then
    echo "✅ 发现空闲GPU: $available_gpus"
    
    # 计算GPU数量
    gpu_count_available=$(echo $available_gpus | tr ',' '\n' | wc -l)
    
               # 为了给其他用户留出GPU，最多选择4个GPU
           if [ "$gpu_count_available" -ge 4 ]; then
               # 选择前4个GPU，给其他用户留出1个
               selected_gpus=$(echo $available_gpus | cut -d',' -f1-4)
               selection_reason="选择4个GPU，给其他用户留出1个GPU"
           else
               selected_gpus=$available_gpus
               selection_reason="使用所有可用空闲GPU"
           fi
else
    echo "⚠️  没有完全空闲的GPU，选择最合适的GPU组合..."
    
    # 重新解析以获取GPU分数
    declare -A gpu_scores
    
    while IFS= read -r line; do
        if [ -z "$line" ]; then
            continue
        fi
        
        if [[ $line =~ \[([0-9]+)\].*\|[[:space:]]*([0-9]+)°C,[[:space:]]*([0-9]+)[[:space:]]*%[[:space:]]*\|[[:space:]]*([0-9]+)[[:space:]]*/[[:space:]]*([0-9]+)[[:space:]]*MB ]]; then
            gpu_id="${BASH_REMATCH[1]}"
            utilization="${BASH_REMATCH[3]}"
            memory_used="${BASH_REMATCH[4]}"
            memory_total="${BASH_REMATCH[5]}"
            
            # 计算内存使用率
            memory_usage_percent=$((memory_used * 100 / memory_total))
            
            # 计算GPU分数 (内存使用率 + 利用率)
            gpu_score=$((memory_usage_percent + utilization))
            gpu_scores[$gpu_id]=$gpu_score
        fi
    done <<< "$gpustat_output"
    
    # 选择两个分数最低的GPU
    best_gpus=""
    best_score=1000000
    
    # 遍历所有可能的GPU组合
    for gpu1 in 0 1 2 3 4; do
        for gpu2 in 0 1 2 3 4; do
            if [ "$gpu1" -lt "$gpu2" ]; then
                # 获取GPU分数
                score1=${gpu_scores[$gpu1]:-0}
                score2=${gpu_scores[$gpu2]:-0}
                total_score=$((score1 + score2))
                
                # 检查是否比当前最佳组合更好
                if [ "$total_score" -lt "$best_score" ]; then
                    best_score=$total_score
                    best_gpus="$gpu1,$gpu2"
                fi
            fi
        done
    done
    
    selected_gpus=$best_gpus
    selection_reason="选择负载最低的GPU组合 (总分: $best_score)"
fi

echo ""
echo "🎯 最终选择:"
echo "   GPU: $selected_gpus"
echo "   原因: $selection_reason"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=$selected_gpus
export SELECTED_GPUS=$selected_gpus

# 确保环境变量在当前shell中生效
echo "export CUDA_VISIBLE_DEVICES=$selected_gpus" >> /tmp/gpu_env.sh
echo "export SELECTED_GPUS=$selected_gpus" >> /tmp/gpu_env.sh

echo ""
echo "🚀 环境变量已设置:"
echo "   CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "   SELECTED_GPUS=$SELECTED_GPUS"

# 返回选择的GPU数量
gpu_count=$(echo $selected_gpus | tr ',' '\n' | wc -l)
echo "   GPU数量: $gpu_count"

# 将结果保存到文件供其他脚本使用
echo "$selected_gpus" > /tmp/selected_gpus.txt
echo "$gpu_count" > /tmp/gpu_count.txt
echo "$selection_reason" > /tmp/selection_reason.txt

echo ""
echo "✅ GPU选择完成！"
