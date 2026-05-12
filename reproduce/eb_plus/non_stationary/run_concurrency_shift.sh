#!/bin/bash

# Concurrency Shift 实验脚本
# 验证 THETA 的 IFR 在线 controller 在并发突变时的行为:
#   1. θ* 能在并发变化后快速收敛到新最优值
#   2. 系统保持 memory-safe (无 OOM)
#   3. 吞吐量能跟随并发变化
#
# 实验设计 (默认 3 phases):
#   Phase 1: 低并发   (concurrency=32)
#   Phase 2: 高并发   (concurrency=2048)
#   Phase 3: 中并发   (concurrency=500)
#   Server 保持运行，顺序发送不同并发的 benchmark
#   对比: pd_ifr (自适应 θ*) vs pd_ratio (固定 θ*=0.8)
#
# 用法: ./run_concurrency_shift.sh [GPU_ID]
#
# 环境变量:
#   MODEL: 模型路径，默认 Qwen/Qwen3-8B
#   NUM_PROMPTS_PER_PHASE: 每个阶段的请求数，默认 2000
#   CONCURRENCY_PHASES: 并发阶段，格式 "concurrency[:num_prompts],..."
#                      例: "32:500,2048:4000,500:2000" 或 "32,2048,500" (用默认数量)
#   INPUT_LEN: 固定 input 长度，默认 512
#   OUTPUT_LEN: 固定 output 长度，默认 256
#   IFR_WINDOW_SIZE: IFR 滑动窗口大小，默认 500

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../common/common.sh"

# 实验参数
GPU_ID=${1:-0}
MODEL=${MODEL:-"Qwen/Qwen3-8B"}
MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||')
NUM_PROMPTS_PER_PHASE=${NUM_PROMPTS_PER_PHASE:-2000}
CONCURRENCY_PHASES=${CONCURRENCY_PHASES:-"32:500,2048:4000,500:2000"}
INPUT_LEN=${INPUT_LEN:-512}
OUTPUT_LEN=${OUTPUT_LEN:-256}
OUTPUT_VARIANCE=${OUTPUT_VARIANCE:-0.25}
K_RATIO=${K_RATIO:-0.8}
BASE_PORT=${BASE_PORT:-14000}
IFR_WINDOW_SIZE=${IFR_WINDOW_SIZE:-500}
SOURCE_DATASET=${SOURCE_DATASET:-"alpaca"}

# 最优配置 (H200)
TB=${TB:-18432}
BS=${BS:-2048}

# 解析并发阶段 (格式: concurrency[:num_prompts],...)
IFS=',' read -ra _RAW_PHASES <<< "$CONCURRENCY_PHASES"
NUM_PHASES=${#_RAW_PHASES[@]}
PHASE_CONCURRENCIES=()
PHASE_NUM_PROMPTS=()
MAX_PHASE_PROMPTS=0
for _p in "${_RAW_PHASES[@]}"; do
    _p=$(echo "$_p" | tr -d ' ')
    if [[ "$_p" == *:* ]]; then
        PHASE_CONCURRENCIES+=("${_p%%:*}")
        PHASE_NUM_PROMPTS+=("${_p##*:}")
    else
        PHASE_CONCURRENCIES+=("$_p")
        PHASE_NUM_PROMPTS+=("$NUM_PROMPTS_PER_PHASE")
    fi
    local_n=${PHASE_NUM_PROMPTS[-1]}
    [ "$local_n" -gt "$MAX_PHASE_PROMPTS" ] && MAX_PHASE_PROMPTS=$local_n
done

# 硬件校准文件 (不存在则自动运行校准)
ensure_calibration "$MODEL" "$MODEL_SHORT"

# 输出目录
OUTPUT_DIR="${SCRIPT_DIR}/../outputs/concurrency_shift_${MODEL_SHORT}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR/logs"

# 初始化环境
init_experiment_env

echo "========================================"
echo "Concurrency Shift 实验 (${NUM_PHASES}-phase)"
echo "========================================"
echo ""
echo "实验配置:"
echo "  MODEL: $MODEL"
echo "  GPU: $GPU_ID"
echo "  TB: $TB, BS: $BS"
echo "  NUM_PROMPTS_PER_PHASE: $NUM_PROMPTS_PER_PHASE"
echo "  CONCURRENCY_PHASES: $CONCURRENCY_PHASES"
echo "  INPUT_LEN: $INPUT_LEN, OUTPUT_LEN: $OUTPUT_LEN"
echo "  IFR_WINDOW_SIZE: $IFR_WINDOW_SIZE"
echo ""

# ========================================
# Step 1: 生成统一分布的合成数据集
# ========================================
# 每个 phase 复用同一批 prompts，只改变并发度
SYNTHETIC_DATASET="${OUTPUT_DIR}/synthetic_uniform.jsonl"
echo "生成合成数据集 (uniform: input~${INPUT_LEN}, output~${OUTPUT_LEN})..."

python3 "${SCRIPT_DIR}/generate_distribution_shift_dataset.py" \
    --model "$MODEL" \
    --num-prompts-per-phase "$MAX_PHASE_PROMPTS" \
    --phases "${INPUT_LEN}:${OUTPUT_LEN}" \
    --variance "$OUTPUT_VARIANCE" \
    --source-dataset "$SOURCE_DATASET" \
    --output "$SYNTHETIC_DATASET" \
    --seed 42

echo ""
echo "数据集已生成: $SYNTHETIC_DATASET"

# 构建 concurrency phases JSON array
PHASES_JSON=$(python3 -c "
import json
concurrencies = '${PHASE_CONCURRENCIES[*]}'.split()
num_prompts = '${PHASE_NUM_PROMPTS[*]}'.split()
result = [{'concurrency': int(c), 'num_prompts': int(n)} for c, n in zip(concurrencies, num_prompts)]
print(json.dumps(result))
")

# 保存实验配置
cat > "${OUTPUT_DIR}/experiment_config.json" << EOF
{
    "experiment_type": "concurrency_shift",
    "purpose": "Validate IFR controller adaptation under concurrency level changes",
    "model": "${MODEL}",
    "gpu_id": ${GPU_ID},
    "tb": ${TB},
    "bs": ${BS},
    "default_num_prompts_per_phase": ${NUM_PROMPTS_PER_PHASE},
    "num_phases": ${NUM_PHASES},
    "concurrency_phases": ${PHASES_JSON},
    "input_len": ${INPUT_LEN},
    "output_len": ${OUTPUT_LEN},
    "output_variance": ${OUTPUT_VARIANCE},
    "ifr_window_size": ${IFR_WINDOW_SIZE},
    "k_ratio": ${K_RATIO},
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
            export VLLM_PD_K_MODE=ratio
            export VLLM_PD_K_RATIO=$K_RATIO
            ;;
    esac

    wait_for_gpu_memory $GPU_ID 60 || return 1

    # 启动服务 (整个实验期间保持运行)
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

    # 顺序运行每个并发阶段 (server 保持运行，IFR 状态连续)
    local phase_idx=0
    local overall_status=0

    for phase_idx_0 in $(seq 0 $((NUM_PHASES - 1))); do
        local concurrency=${PHASE_CONCURRENCIES[$phase_idx_0]}
        local phase_prompts=${PHASE_NUM_PROMPTS[$phase_idx_0]}
        phase_idx=$((phase_idx_0 + 1))

        echo ""
        echo "--- Phase ${phase_idx}/${NUM_PHASES}: concurrency=${concurrency}, num_prompts=${phase_prompts} ---"

        local bench_status=0
        vllm bench serve \
            --model "$MODEL" \
            --base-url "http://localhost:${port}" \
            --dataset-name custom \
            --dataset-path "$SYNTHETIC_DATASET" \
            --custom-output-len -1 \
            --ignore-eos \
            --num-prompts "$phase_prompts" \
            --num-warmups 0 \
            --request-rate inf \
            --max-concurrency "$concurrency" \
            --save-result \
            --save-detailed \
            --result-dir "${OUTPUT_DIR}" \
            --result-filename "bench_${scheduler}_phase${phase_idx}_c${concurrency}.json" \
            >> "$log_file" 2>&1 || bench_status=$?

        if [ $bench_status -eq 0 ]; then
            echo "Phase ${phase_idx} 完成 (concurrency=${concurrency}, prompts=${phase_prompts})"
        else
            echo "Phase ${phase_idx} 失败 (concurrency=${concurrency}, exit=$bench_status)"
            overall_status=$bench_status
        fi
    done

    kill_server $server_pid $GPU_ID

    if [ $overall_status -eq 0 ]; then
        echo "完成: ${scheduler}"
    else
        echo "部分失败: ${scheduler}"
    fi

    return $overall_status
}

# 运行指定的 scheduler (可通过 SCHEDULERS 环境变量控制)
# 例: SCHEDULERS="baseline,pd_auto" bash run_concurrency_shift.sh 0
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
echo "  python reproduce/eb_plus/non_stationary/plot_distribution_shift.py $OUTPUT_DIR"
