#!/bin/bash

# ==================================================
# 自动化运行 Model Collapse 消融实验套件
# ==================================================

# 1. 定义要遍历的策略数组
STRATEGIES=("micro_only" "macro_only")

# 定义你的 Python 脚本名称
PYTHON_SCRIPT="/home/ubuntu/data/simc/gpt2_wikitext2/rejection_sampling_v2.py"

# 创建一个专门的日志目录，用来存放控制台输出
LOG_DIR="/home/ubuntu/data/simc/gpt2_wikitext2/rejection_sampling_results_v6/experiment_logs"
mkdir -p ${LOG_DIR}

echo "=================================================="
echo "🚀 启动 Model Collapse 拒绝采样自动化实验"
echo "共包含 ${#STRATEGIES[@]} 组策略: ${STRATEGIES[*]}"
echo "=================================================="

# 2. 循环遍历所有策略
for strategy in "${STRATEGIES[@]}"; do
    echo ""
    echo ">>> [$(date +'%Y-%m-%d %H:%M:%S')] 开始运行策略: ${strategy} <<<"
    echo "--------------------------------------------------"
    
    # 日志文件路径
    LOG_FILE="${LOG_DIR}/run_${strategy}.log"
    
    # 3. 执行 Python 脚本，并将输出同时打印到终端和写入日志文件 (使用 tee)
    # 如果你不想在终端看到大量输出，可以把 2>&1 | tee ${LOG_FILE} 换成 > ${LOG_FILE} 2>&1
    python ${PYTHON_SCRIPT} --strategy ${strategy} 2>&1 | tee ${LOG_FILE}
    
    # 4. 检查上一条命令的退出状态码
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo ">>> [成功] 策略 '${strategy}' 运行完毕! 日志已保存至 ${LOG_FILE}"
    else
        echo ">>> [错误] 策略 '${strategy}' 运行失败，请检查日志 ${LOG_FILE}!"
        # 如果某一组失败了，停止整个脚本以防止浪费算力
        echo ">>> 实验意外终止。"
        exit 1
    fi
done

echo ""
echo "=================================================="
echo "🎉 所有实验组已全部成功运行完毕！"
echo "=================================================="