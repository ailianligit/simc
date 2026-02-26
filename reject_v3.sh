#!/bin/bash

# ==================================================
# 补充运行 dynamic 模式下的 nll_fbd 和 fbd_only
# ==================================================

# 1. 定义成对的策略配置 (必须一一对应)
TEMP_STRATEGIES=('dynamic' 'dynamic' 'fixed')
FILTER_STRATEGIES=('nll_fbd' 'fbd_only' 'fbd_only')

# 确保脚本路径正确
PYTHON_SCRIPT="/home/ubuntu/data/simc/gpt2_wikitext2/rejection_sampling_v3.py" 
LOG_DIR="/home/ubuntu/data/simc/gpt2_wikitext2/rejection_sampling_results_v9/experiment_logs"
mkdir -p ${LOG_DIR}

echo "=================================================="
echo "🚀 启动 Model Collapse 补充实验"
echo "=================================================="

# 2. 使用索引循环遍历配对
for i in "${!TEMP_STRATEGIES[@]}"; do
    temp="${TEMP_STRATEGIES[$i]}"
    filter="${FILTER_STRATEGIES[$i]}"
    group_name="${temp}_${filter}"
    
    echo ""
    echo ">>> [$(date +'%Y-%m-%d %H:%M:%S')] 开始运行组别: ${group_name} (${expr $i + 1}/${#TEMP_STRATEGIES[@]}) <<<"
    echo "--------------------------------------------------"
    
    LOG_FILE="${LOG_DIR}/run_${group_name}.log"
    
    python ${PYTHON_SCRIPT} \
        --temp_strategy ${temp} \
        --filter_strategy ${filter} \
        2>&1 | tee ${LOG_FILE}
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo ">>> [成功] 组别 '${group_name}' 运行完毕! 日志: ${LOG_FILE}"
    else
        echo ">>> [错误] 组别 '${group_name}' 运行失败!"
        echo ">>> 实验意外终止。"
        exit 1
    fi
done

echo ""
echo "🎉 补充实验组成功运行完毕！"