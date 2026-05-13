#!/bin/bash

# 一键运行所有实验脚本
# 用法: ./run_all_experiments.sh <MODEL> [MAX_GPUS]
#
# 示例:
#   ./reproduce/run_all_experiments.sh Qwen/Qwen3-8B 4
#   ./reproduce/run_all_experiments.sh meta-llama/Llama-3.1-8B 2

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 检查参数
if [ -z "${1:-}" ]; then
    echo "用法: $0 <MODEL> [MAX_GPUS]"
    echo ""
    echo "参数:"
    echo "  MODEL      模型名称 (如 Qwen/Qwen3-8B)"
    echo "  MAX_GPUS   使用的 GPU 数量 (默认 4)"
    echo ""
    echo "示例:"
    echo "  $0 Qwen/Qwen3-8B 4"
    echo "  $0 meta-llama/Llama-3.1-8B 2"
    echo ""
    echo "环境变量 (可选):"
    echo "  SKIP_CALIBRATION=true   跳过校准步骤"
    echo "  SKIP_EXPORT=true        跳过数据集导出步骤"
    echo "  EXPERIMENTS=\"sharegpt numina_math\"  只运行指定实验"
    echo "  SCHEDULERS=\"baseline pd_ratio pd_ifr\"  只运行指定调度器模式"
    echo "  SKIP_EXISTING=0         不跳过已有结果 (默认跳过)"
    echo "  VERSION=1               文件名后缀，用于重复运行生成 ifr_1, ifr_2 等"
    echo ""
    echo "调度器模式 (SCHEDULERS):"
    echo "  baseline   vLLM 默认调度器"
    echo "  pd_ratio   PD scheduler with ratio mode (θ*=K_RATIO)"
    echo "  pd_ifr     PD scheduler with IFR mode (adaptive θ* based on hazard rate)"
    echo ""
    echo "示例 - 只运行 pd_ifr 模式的实验:"
    echo "  SCHEDULERS=pd_ifr $0 Qwen/Qwen3-8B 4"
    exit 1
fi

MODEL="$1"
MAX_GPUS=${2:-4}
MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||')

echo "========================================"
echo "PD Scheduler 全量实验"
echo "========================================"
echo ""
echo "配置:"
echo "  MODEL: $MODEL"
echo "  MODEL_SHORT: $MODEL_SHORT"
echo "  MAX_GPUS: $MAX_GPUS"
echo ""

# 创建输出目录
mkdir -p "${SCRIPT_DIR}/outputs"

# 记录开始时间
START_TIME=$(date +%s)
log_time() {
    local now=$(date +%s)
    local elapsed=$((now - START_TIME))
    local hours=$((elapsed / 3600))
    local minutes=$(((elapsed % 3600) / 60))
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [${hours}h ${minutes}m] $1"
}

# ============================================================
# Step 0: 硬件校准
# ============================================================
CALIBRATION_FILE="${SCRIPT_DIR}/outputs/pd_calibration_${MODEL_SHORT}.json"

if [ "${SKIP_CALIBRATION:-false}" = "true" ]; then
    log_time "跳过校准步骤 (SKIP_CALIBRATION=true)"
elif [ -f "$CALIBRATION_FILE" ]; then
    log_time "校准文件已存在: $CALIBRATION_FILE"
    echo "  如需重新校准，请删除此文件或设置 SKIP_CALIBRATION=false"
else
    log_time "开始硬件校准..."
    # 某些模型不支持 float16，可通过 DTYPE=bfloat16 指定
    DTYPE_ARG=""
    if [ -n "${DTYPE:-}" ]; then
        DTYPE_ARG="--dtype $DTYPE"
    fi
    python -m vllm.v1.core.sched.calibration --model "$MODEL" $DTYPE_ARG --output "$CALIBRATION_FILE"
    log_time "校准完成: $CALIBRATION_FILE"
fi

# 验证校准文件
if [ ! -f "$CALIBRATION_FILE" ]; then
    echo "错误: 校准文件不存在: $CALIBRATION_FILE"
    exit 1
fi

export VLLM_PD_CALIBRATION_FILE="$CALIBRATION_FILE"

# SKIP_EXISTING: 默认跳过已有结果文件 (方便只运行某个 scheduler 模式)
export SKIP_EXISTING=${SKIP_EXISTING:-1}
echo "  SKIP_EXISTING: $SKIP_EXISTING"
echo ""

# ============================================================
# Step 1: 导出数据集
# ============================================================
log_time "准备数据集..."

SHAREGPT_PROMPTS="${SCRIPT_DIR}/outputs/sharegpt_prompts.jsonl"
NUMINA_PROMPTS="${SCRIPT_DIR}/outputs/numina_math_prompts.jsonl"
LONGBENCH_PROMPTS="${SCRIPT_DIR}/outputs/longbench_prefill.jsonl"
WILDCHAT_DATA="${SCRIPT_DIR}/outputs/wildchat_multiturn.json"

if [ "${SKIP_EXPORT:-false}" = "true" ]; then
    log_time "跳过数据集导出 (SKIP_EXPORT=true)"
else
    # ShareGPT
    if [ ! -f "$SHAREGPT_PROMPTS" ]; then
        log_time "导出 ShareGPT 数据集..."
        # 下载原始数据
        if [ ! -f "ShareGPT_V3_unfiltered_cleaned_split.json" ]; then
            wget -q https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
        fi
        python "${SCRIPT_DIR}/common/export_dataset.py" \
            --dataset sharegpt \
            --model "$MODEL" \
            --num-samples 4000 \
            --output "$SHAREGPT_PROMPTS"
        rm -f ShareGPT_V3_unfiltered_cleaned_split.json
    else
        log_time "ShareGPT 数据集已存在"
    fi

    # numina_math
    if [ ! -f "$NUMINA_PROMPTS" ]; then
        log_time "导出 numina_math 数据集..."
        python "${SCRIPT_DIR}/common/export_dataset.py" \
            --dataset numina_math \
            --model "$MODEL" \
            --num-samples 4000 \
            --min-output-len 800 \
            --output "$NUMINA_PROMPTS"
    else
        log_time "numina_math 数据集已存在"
    fi

    # longbench
    if [ ! -f "$LONGBENCH_PROMPTS" ]; then
        log_time "导出 longbench 数据集..."
        python "${SCRIPT_DIR}/common/export_dataset.py" \
            --dataset longbench \
            --model "$MODEL" \
            --num-samples 4000 \
            --min-input-len 1000 \
            --max-input-len 4000 \
            --output "$LONGBENCH_PROMPTS"
    else
        log_time "longbench 数据集已存在"
    fi

    # WildChat (多轮对话)
    if [ ! -f "$WILDCHAT_DATA" ]; then
        log_time "导出 WildChat 多轮对话数据集..."
        python "${SCRIPT_DIR}/real_workloads/multiturn/export_dataset.py" \
            --dataset wildchat \
            --model "$MODEL" \
            --num-conversations 3000 \
            --min-turns 6 \
            --output "$WILDCHAT_DATA"
    else
        log_time "WildChat 数据集已存在"
    fi
fi

echo ""

# ============================================================
# Step 2: 运行实验
# ============================================================

# 默认运行所有实验
EXPERIMENTS=${EXPERIMENTS:-"sharegpt numina_math longbench wildchat"}

run_experiment() {
    local name=$1
    local dataset=$2
    local output_len=$3
    local enable_thinking=$4
    local script=$5

    log_time "开始实验: $name"
    echo "  数据集: $dataset"
    echo "  输出长度: $output_len"
    echo "  Thinking: $enable_thinking"
    echo ""

    ENABLE_THINKING=$enable_thinking \
    CUSTOM_OUTPUT_LEN=$output_len \
    MODEL=$MODEL \
    DTYPE=${DTYPE:-} \
        "$script" "$dataset" "$MAX_GPUS"

    log_time "实验完成: $name"
    echo ""
}

# ShareGPT: balanced workload
if [[ "$EXPERIMENTS" == *"sharegpt"* ]]; then
    if [ -f "$SHAREGPT_PROMPTS" ]; then
        run_experiment "ShareGPT" "$SHAREGPT_PROMPTS" 500 false "${SCRIPT_DIR}/real_workloads/run_grid_search.sh"
    else
        log_time "跳过 ShareGPT (数据集不存在)"
    fi
fi

# numina_math: decode-heavy
if [[ "$EXPERIMENTS" == *"numina_math"* ]]; then
    if [ -f "$NUMINA_PROMPTS" ]; then
        run_experiment "numina_math" "$NUMINA_PROMPTS" 4000 true "${SCRIPT_DIR}/real_workloads/run_grid_search.sh"
    else
        log_time "跳过 numina_math (数据集不存在)"
    fi
fi

# longbench: prefill-heavy
if [[ "$EXPERIMENTS" == *"longbench"* ]]; then
    if [ -f "$LONGBENCH_PROMPTS" ]; then
        run_experiment "longbench" "$LONGBENCH_PROMPTS" 20 false "${SCRIPT_DIR}/real_workloads/run_grid_search.sh"
    else
        log_time "跳过 longbench (数据集不存在)"
    fi
fi

# WildChat: 多轮对话 (prefix cache)
if [[ "$EXPERIMENTS" == *"wildchat"* ]]; then
    if [ -f "$WILDCHAT_DATA" ]; then
        log_time "开始实验: WildChat (多轮对话)"
        echo "  数据集: $WILDCHAT_DATA"
        echo ""

        MODEL=$MODEL \
        DTYPE=${DTYPE:-} \
            "${SCRIPT_DIR}/real_workloads/multiturn/run_benchmark.sh" "$WILDCHAT_DATA" "$MAX_GPUS"

        log_time "实验完成: WildChat"
        echo ""
    else
        log_time "跳过 WildChat (数据集不存在)"
    fi
fi

# ============================================================
# Step 3: 汇总结果
# ============================================================
echo ""
echo "========================================"
log_time "所有实验完成!"
echo "========================================"
echo ""
echo "结果目录:"

# 列出生成的结果目录
for dir in "${SCRIPT_DIR}/outputs/grid_search_"*"_${MODEL_SHORT}_"* "${SCRIPT_DIR}/outputs/multiturn_"*"_${MODEL_SHORT}_"*; do
    if [ -d "$dir" ]; then
        echo "  $dir"
    fi
done

echo ""
echo "运行分析脚本:"
echo ""

# ShareGPT
SHAREGPT_DIR=$(ls -d "${SCRIPT_DIR}/outputs/grid_search_sharegpt_prompts_${MODEL_SHORT}_"* 2>/dev/null | head -1)
if [ -n "$SHAREGPT_DIR" ] && [ -d "$SHAREGPT_DIR" ]; then
    echo "# ShareGPT"
    echo "python ${SCRIPT_DIR}/real/analyze_grid_search.py $SHAREGPT_DIR"
    echo ""
fi

# numina_math
NUMINA_DIR=$(ls -d "${SCRIPT_DIR}/outputs/grid_search_numina_math_prompts_${MODEL_SHORT}_"* 2>/dev/null | head -1)
if [ -n "$NUMINA_DIR" ] && [ -d "$NUMINA_DIR" ]; then
    echo "# numina_math"
    echo "python ${SCRIPT_DIR}/real/analyze_grid_search.py $NUMINA_DIR"
    echo ""
fi

# longbench
LONGBENCH_DIR=$(ls -d "${SCRIPT_DIR}/outputs/grid_search_longbench_prefill_${MODEL_SHORT}_"* 2>/dev/null | head -1)
if [ -n "$LONGBENCH_DIR" ] && [ -d "$LONGBENCH_DIR" ]; then
    echo "# longbench"
    echo "python ${SCRIPT_DIR}/real/analyze_grid_search.py $LONGBENCH_DIR"
    echo ""
fi

# WildChat
WILDCHAT_DIR=$(ls -d "${SCRIPT_DIR}/outputs/multiturn_wildchat_multiturn_${MODEL_SHORT}_"* 2>/dev/null | head -1)
if [ -n "$WILDCHAT_DIR" ] && [ -d "$WILDCHAT_DIR" ]; then
    echo "# WildChat (多轮对话)"
    echo "python ${SCRIPT_DIR}/multiturn/analyze_results.py $WILDCHAT_DIR"
    echo ""
fi

# 计算总耗时
END_TIME=$(date +%s)
TOTAL_ELAPSED=$((END_TIME - START_TIME))
TOTAL_HOURS=$((TOTAL_ELAPSED / 3600))
TOTAL_MINUTES=$(((TOTAL_ELAPSED % 3600) / 60))
echo "========================================"
echo "总耗时: ${TOTAL_HOURS}h ${TOTAL_MINUTES}m"
echo "========================================"
