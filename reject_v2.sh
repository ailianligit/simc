#!/bin/bash

# 设置遇到错误时立即退出脚本
set -e

# 定义要运行的 Python 脚本名称
PYTHON_SCRIPT="/home/ubuntu/data/simc/gpt2_wikitext2/rejection_sampling_v2.py"

# 定义需要遍历的策略数组
STRATEGIES=("combined" "baseline" "micro_only" "macro_only")

echo "=================================================================="
echo "  Starting Automated Model Collapse Rejection Experiments"
echo "  Total strategies to evaluate: ${#STRATEGIES[@]}"
echo "=================================================================="

# 循环遍历并执行
for STRATEGY in "${STRATEGIES[@]}"; do
    echo ""
    echo "------------------------------------------------------------------"
    echo ">>> Current Running Strategy: [ $STRATEGY ]"
    echo ">>> Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "------------------------------------------------------------------"
    
    # 执行 Python 脚本，并传入当前的 strategy 参数
    # 如果你想把日志输出到文件而不是终端，可以在下面这行末尾加上: 2>&1 | tee "logs_${STRATEGY}.log"
    python $PYTHON_SCRIPT --strategy "$STRATEGY"
    
    echo ">>> Finished Strategy: [ $STRATEGY ]"
    
    # 可选：在每组实验之间强制清理一下系统缓存和显存（需系统权限）
    # sync; echo 3 > /proc/sys/vm/drop_caches
done

echo ""
echo "=================================================================="
echo "  All experiments completed successfully!"
echo "  Check the './rejection_results/' directory for JSON metrics."
echo "=================================================================="