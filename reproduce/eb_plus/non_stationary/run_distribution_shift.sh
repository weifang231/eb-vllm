#!/bin/bash

# Distribution Shift 实验脚本 (3-phase synthetic data)
# 验证 THETA 的 IFR 在线 controller 在 workload 突变时的行为:
#   1. θ* 能在 ~W 个样本内收敛到新最优值
#   2. 系统保持 memory-safe (无 OOM)
#   3. 吞吐量暂时下降幅度有限
#
# 实验设计 (3 phases):
#   Phase 1: prefill-heavy  (input~1024, output~128)
#   Phase 2: balanced        (input~512,  output~512)
#   Phase 3: decode-heavy   (input~128,  output~1024)
#   对比: pd_ifr (自适应 θ*) vs pd_ratio (固定 θ*=0.8)
#
# 用法: ./run_distribution_shift.sh [GPU_ID]
#
# 环境变量:
#   MODEL: 模型路径，默认 Qwen/Qwen3-8B
#   NUM_PROMPTS_PER_PHASE: 每个阶段的请求数，默认 2000
#   MAX_CONCURRENCY: 最大并发，默认 2048
#   IFR_WINDOW_SIZE: IFR 滑动窗口大小，默认 500
#   PHASES: phase 定义，默认 "1024:128,512:512,128:1024"
#   OUTPUT_VARIANCE: output_len 方差比例，默认 0.25

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../common/common.sh"

# 实验参数
GPU_ID=${1:-0}
MODEL=${MODEL:-"Qwen/Qwen3-8B"}
MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||')
NUM_PROMPTS_PER_PHASE=${NUM_PROMPTS_PER_PHASE:-2000}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-2048}
K_RATIO=${K_RATIO:-0.8}
BASE_PORT=${BASE_PORT:-13000}
IFR_WINDOW_SIZE=${IFR_WINDOW_SIZE:-500}
PHASES=${PHASES:-"1024:128,512:512,128:1024"}
OUTPUT_VARIANCE=${OUTPUT_VARIANCE:-0.25}
SOURCE_DATASET=${SOURCE_DATASET:-"alpaca"}

# 最优配置 (H200)
TB=${TB:-18432}
BS=${BS:-2048}

# 计算 phase 数量和总请求数
NUM_PHASES=$(echo "$PHASES" | tr ',' '\n' | wc -l)
TOTAL_PROMPTS=$((NUM_PROMPTS_PER_PHASE * NUM_PHASES))

# 硬件校准文件 (不存在则自动运行校准)
ensure_calibration "$MODEL" "$MODEL_SHORT"

# 输出目录
OUTPUT_DIR="${SCRIPT_DIR}/../outputs/distribution_shift_${MODEL_SHORT}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR/logs"

# 初始化环境
init_experiment_env

echo "========================================"
echo "Distribution Shift 实验 (${NUM_PHASES}-phase)"
echo "========================================"
echo ""
echo "实验配置:"
echo "  MODEL: $MODEL"
echo "  GPU: $GPU_ID"
echo "  TB: $TB, BS: $BS"
echo "  NUM_PROMPTS_PER_PHASE: $NUM_PROMPTS_PER_PHASE"
echo "  TOTAL_PROMPTS: $TOTAL_PROMPTS"
echo "  MAX_CONCURRENCY: $MAX_CONCURRENCY"
echo "  IFR_WINDOW_SIZE: $IFR_WINDOW_SIZE"
echo "  PHASES: $PHASES"
echo "  OUTPUT_VARIANCE: $OUTPUT_VARIANCE"
echo ""

# ========================================
# Step 1: 生成合成数据集
# ========================================
SYNTHETIC_DATASET="${OUTPUT_DIR}/synthetic_${NUM_PHASES}phase.jsonl"
echo "生成合成数据集..."

python3 "${SCRIPT_DIR}/generate_distribution_shift_dataset.py" \
    --model "$MODEL" \
    --num-prompts-per-phase "$NUM_PROMPTS_PER_PHASE" \
    --phases "$PHASES" \
    --variance "$OUTPUT_VARIANCE" \
    --source-dataset "$SOURCE_DATASET" \
    --output "$SYNTHETIC_DATASET" \
    --seed 42

echo ""
echo "数据集已生成: $SYNTHETIC_DATASET"

# 构建 phases JSON array
PHASES_JSON=$(python3 -c "
import json
phases = '$PHASES'.split(',')
result = []
for p in phases:
    inp, out = p.strip().split(':')
    inp, out = int(inp), int(out)
    ratio = out / max(inp, 1)
    if ratio > 1.5:
        name = 'decode-heavy'
    elif ratio < 0.5:
        name = 'prefill-heavy'
    else:
        name = 'balanced'
    result.append({'name': name, 'input_mean': inp, 'output_mean': out})
print(json.dumps(result))
")

# 保存实验配置
cat > "${OUTPUT_DIR}/experiment_config.json" << EOF
{
    "experiment_type": "distribution_shift",
    "purpose": "Validate IFR controller convergence under workload distribution shift",
    "model": "${MODEL}",
    "gpu_id": ${GPU_ID},
    "tb": ${TB},
    "bs": ${BS},
    "num_prompts_per_phase": ${NUM_PROMPTS_PER_PHASE},
    "total_prompts": ${TOTAL_PROMPTS},
    "num_phases": ${NUM_PHASES},
    "phases": ${PHASES_JSON},
    "max_concurrency": ${MAX_CONCURRENCY},
    "ifr_window_size": ${IFR_WINDOW_SIZE},
    "k_ratio": ${K_RATIO},
    "output_variance": ${OUTPUT_VARIANCE},
    "schedulers": ["baseline", "pd_ifr", "pd_ratio", "pd_auto"],
    "calibration_file": "${VLLM_PD_CALIBRATION_FILE}",
    "timestamp": "$(date -Iseconds)"
}
EOF

# ========================================
# Step 2: 运行实验
# ========================================
run_single_experiment() {
    local scheduler=$1
    local port=$((BASE_PORT))
    local log_file="${OUTPUT_DIR}/logs/${scheduler}.log"

    echo ""
    echo "========================================"
    echo "运行: ${scheduler}"
    echo "========================================"

    : > "$log_file"

    # 设置环境变量
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    export VLLM_COLLECT_SCHEDULE_STATS=1

    # Clean all PD env vars first, then set only what's needed
    unset VLLM_USE_PD_SCHEDULER VLLM_PD_K_MODE VLLM_PD_K_RATIO \
          VLLM_PD_K_STAR VLLM_PD_IFR_WINDOW_SIZE VLLM_PD_SCHEDULER_MODE

    case "$scheduler" in
        baseline)
            ;;
        pd_ifr)
            export VLLM_USE_PD_SCHEDULER=1
            export VLLM_PD_K_MODE=ifr
            export VLLM_PD_IFR_WINDOW_SIZE=$IFR_WINDOW_SIZE
            ;;
        pd_ratio)
            export VLLM_USE_PD_SCHEDULER=1
            export VLLM_PD_K_MODE=ratio
            export VLLM_PD_K_RATIO=$K_RATIO
            ;;
        pd_auto)
            export VLLM_PD_SCHEDULER_MODE=auto
            export VLLM_PD_K_MODE=ifr
            export VLLM_PD_IFR_WINDOW_SIZE=$IFR_WINDOW_SIZE
            ;;
    esac

    wait_for_gpu_memory $GPU_ID 60 || return 1

    # 启动服务
    local dtype_arg=""
    if [ -n "${DTYPE:-}" ]; then
        dtype_arg="--dtype $DTYPE"
    fi

    VLLM_SCHEDULE_STATS_FILE="${OUTPUT_DIR}/${scheduler}_stats.json" \
    vllm serve "$MODEL" \
        --port "$port" \
        --gpu-memory-utilization 0.9 \
        --max-num-seqs "$BS" \
        --max-num-batched-tokens "$TB" \
        $dtype_arg >> "$log_file" 2>&1 &
    local server_pid=$!

    if ! wait_for_server $port $server_pid 180 "$log_file"; then
        echo "服务启动失败: ${scheduler}"
        kill_server $server_pid $GPU_ID
        return 1
    fi

    echo "服务已启动 (PID: $server_pid, port: $port)"
    echo "开始 benchmark..."

    # 运行 benchmark
    # --custom-output-len -1: 使用 JSONL 中每个请求的 output_len
    # --ignore-eos: 强制生成指定长度（否则模型遇到 EOS 就停止，无法控制输出长度）
    local bench_status=0
    vllm bench serve \
        --model "$MODEL" \
        --base-url "http://localhost:${port}" \
        --dataset-name custom \
        --dataset-path "$SYNTHETIC_DATASET" \
        --custom-output-len -1 \
        --ignore-eos \
        --num-prompts "$TOTAL_PROMPTS" \
        --num-warmups 0 \
        --request-rate inf \
        --max-concurrency "$MAX_CONCURRENCY" \
        --save-result \
        --save-detailed \
        --result-dir "${OUTPUT_DIR}" \
        --result-filename "bench_${scheduler}.json" \
        >> "$log_file" 2>&1 || bench_status=$?

    kill_server $server_pid $GPU_ID

    if [ $bench_status -eq 0 ]; then
        echo "完成: ${scheduler}"
    else
        echo "失败: ${scheduler}"
    fi

    return $bench_status
}

# 运行指定的 scheduler (可通过 SCHEDULERS 环境变量控制)
# 例: SCHEDULERS="baseline,pd_auto" bash run_distribution_shift.sh 0
DEFAULT_SCHEDULERS="baseline,pd_ifr,pd_ratio,pd_auto"
IFS=',' read -ra SCHEDULER_LIST <<< "${SCHEDULERS:-$DEFAULT_SCHEDULERS}"
for sched in "${SCHEDULER_LIST[@]}"; do
    sched=$(echo "$sched" | tr -d ' ')
    run_single_experiment "$sched" || echo "警告: ${sched} 实验失败 (exit=$?)"
done

echo ""
echo "========================================"
echo "实验完成!"
echo "========================================"
echo ""
echo "结果目录: $OUTPUT_DIR"
echo ""
echo "运行分析脚本:"
echo "  python reproduce/eb_plus/non_stationary/ or long_context/ or disagg/plot_distribution_shift.py $OUTPUT_DIR"
