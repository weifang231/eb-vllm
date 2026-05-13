#!/bin/bash

# 并发度扫描实验脚本 (WildChat 多轮对话)
# 固定每个 scheduler 的最优 (TB, BS) 配置，扫描不同并发度下的 TTFT/TPOT/RPS
#
# 目的:
#   目标 1: 在低并发 (64, 256) 下展示 scheduler 本身的 TTFT 影响 (排队延迟最小化)
#   目标 2: 扫描并发度，找到满足 SLO 约束的最大吞吐
#
# 用法: ./run_concurrency_sweep.sh [MAX_GPUS]
#
# 环境变量:
#   MODEL: 模型路径，默认 Qwen/Qwen3-8B
#   DATASET_PATH: WildChat 数据集路径
#   CONCURRENCY_VALUES: 并发度列表，如 "32 64 128 256 512 1024 2048"
#   SCHEDULERS: 调度器列表，如 "baseline pd_ratio pd_ifr"
#   K_RATIO: PD ratio 模式的 θ*，默认 0.8
#   SKIP_EXISTING: 跳过已有结果，默认 1
#   GPU_TYPE: GPU 类型 (h200/rtx_pro_6000)，用于选择最优配置

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../common/common.sh"

WORKER_PIDS=()
cleanup() {
    for pid in "${WORKER_PIDS[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM HUP

# ========================================
# 最优配置查找表 (来自论文 Table)
# 格式: get_optimal_config <gpu_type> <model_short> <scheduler>
# 返回: TB BS
# ========================================
get_optimal_config() {
    local gpu_type=$1 model_short=$2 scheduler=$3

    # H200 WildChat 最优配置 (来自论文 Table)
    if [ "$gpu_type" = "h200" ]; then
        case "${model_short}|${scheduler}" in
            # EB(1) = pd_ratio
            "Qwen3-8B|pd_ratio")       echo "18432 1536" ;;
            "Qwen3-30B-A3B|pd_ratio")  echo "16384 1024" ;;
            "gemma-3-1b-it|pd_ratio")  echo "16384 1024" ;;
            # CP = baseline
            "Qwen3-8B|baseline")       echo "4096 2048" ;;
            "Qwen3-30B-A3B|baseline")  echo "4096 1536" ;;
            "gemma-3-1b-it|baseline")  echo "18432 256" ;;
            # EB(k̂*) = pd_ifr
            "Qwen3-8B|pd_ifr")        echo "16384 1024" ;;
            "Qwen3-30B-A3B|pd_ifr")   echo "14336 1024" ;;
            "gemma-3-1b-it|pd_ifr")   echo "18432 1536" ;;
            *)
                echo "错误: 未知配置 gpu=${gpu_type} model=${model_short} scheduler=${scheduler}" >&2
                return 1
                ;;
        esac
    elif [ "$gpu_type" = "rtx_pro_6000" ] || [ "$gpu_type" = "a6000" ]; then
        # RTX PRO 6000 / A6000 WildChat 最优配置 (来自论文 Table)
        case "${model_short}|${scheduler}" in
            # EB(1) = pd_ratio
            "Qwen3-8B|pd_ratio")       echo "18432 1024" ;;
            "Qwen3-30B-A3B|pd_ratio")  echo "10240 512" ;;
            "gemma-3-1b-it|pd_ratio")  echo "14336 1536" ;;
            # CP = baseline
            "Qwen3-8B|baseline")       echo "18432 1024" ;;
            "Qwen3-30B-A3B|baseline")  echo "14336 1024" ;;
            "gemma-3-1b-it|baseline")  echo "8192 256" ;;
            # EB(k̂*) = pd_ifr
            "Qwen3-8B|pd_ifr")        echo "10240 1024" ;;
            "Qwen3-30B-A3B|pd_ifr")   echo "18432 512" ;;
            "gemma-3-1b-it|pd_ifr")   echo "14336 2048" ;;
            *)
                echo "错误: 未知配置 gpu=${gpu_type} model=${model_short} scheduler=${scheduler}" >&2
                return 1
                ;;
        esac
    else
        echo "错误: 未知 GPU 类型: ${gpu_type} (支持: h200, rtx_pro_6000, a6000)" >&2
        return 1
    fi
}

# ========================================
# 实验参数
# ========================================
MAX_GPUS=${1:-4}
MODEL=${MODEL:-"Qwen/Qwen3-8B"}
MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||')
GPU_TYPE=${GPU_TYPE:-"h200"}

# 数据集路径
DATASET_PATH=${DATASET_PATH:-"${SCRIPT_DIR}/../outputs/wildchat_multiturn.json"}
if [ ! -f "$DATASET_PATH" ]; then
    echo "错误: 数据集文件不存在: $DATASET_PATH"
    echo "请先导出: python pd_exp/multiturn/export_dataset.py --dataset wildchat --model $MODEL --num-conversations 3000 --min-turns 6 --output $DATASET_PATH"
    exit 1
fi

# 并发度扫描值
if [ -n "${CONCURRENCY_VALUES_STR:-}" ]; then
    read -ra CONCURRENCY_VALUES <<< "$CONCURRENCY_VALUES_STR"
else
    CONCURRENCY_VALUES=(32 64 128 256 512 1024 2048)
fi

# 多轮对话参数
MAX_TURNS=${MAX_TURNS:-12}
LIMIT_MAX_TOKENS=${LIMIT_MAX_TOKENS:-256}
REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-120}
K_RATIO=${K_RATIO:-0.8}
BASE_PORT=${BASE_PORT:-12000}
SCHEDULERS=${SCHEDULERS:-"baseline pd_ratio pd_ifr"}

# 硬件校准文件
if [ -z "${VLLM_PD_CALIBRATION_FILE:-}" ]; then
    DEFAULT_CALIBRATION="${SCRIPT_DIR}/../outputs/pd_calibration_${MODEL_SHORT}.json"
    if [ -f "$DEFAULT_CALIBRATION" ]; then
        export VLLM_PD_CALIBRATION_FILE="$DEFAULT_CALIBRATION"
    else
        echo "错误: 未找到硬件校准文件: $DEFAULT_CALIBRATION"
        echo "请先运行: python -m vllm.v1.core.sched.calibration --model ${MODEL} --output ${DEFAULT_CALIBRATION}"
        exit 1
    fi
fi
echo "使用校准文件: $VLLM_PD_CALIBRATION_FILE"

# 输出目录
OUTPUT_DIR="${SCRIPT_DIR}/../outputs/concurrency_sweep_wildchat_${MODEL_SHORT}_${GPU_TYPE}"
mkdir -p "$OUTPUT_DIR"

# 初始化环境
init_experiment_env

echo "========================================"
echo "并发度扫描实验 (WildChat 多轮对话)"
echo "========================================"

# 检测并选择 GPU
select_gpus $MAX_GPUS

echo ""
echo "实验配置:"
echo "  MODEL: $MODEL"
echo "  GPU_TYPE: $GPU_TYPE"
echo "  DATASET: $DATASET_PATH"
echo "  MAX_TURNS: $MAX_TURNS"
echo "  LIMIT_MAX_TOKENS: $LIMIT_MAX_TOKENS"
echo "  SCHEDULERS: $SCHEDULERS"
echo "  CONCURRENCY_VALUES: ${CONCURRENCY_VALUES[*]}"
echo "  K_RATIO: $K_RATIO"
echo ""

# 打印每个 scheduler 的最优配置
echo "最优配置 (来自论文 Table, GPU=$GPU_TYPE, Workload=WildChat):"
for scheduler in $SCHEDULERS; do
    config=$(get_optimal_config "$GPU_TYPE" "$MODEL_SHORT" "$scheduler")
    read -r tb bs <<< "$config"
    echo "  $scheduler: TB=$tb, BS=$bs"
done
echo ""

# ========================================
# 生成实验队列
# ========================================
QUEUE_FILE="${OUTPUT_DIR}/experiment_queue.txt"
RESUME=${RESUME:-false}

if [ "$RESUME" = "true" ] && [ -f "$QUEUE_FILE" ] && [ -s "$QUEUE_FILE" ]; then
    echo "恢复模式: 使用现有队列文件 ($QUEUE_FILE)"
    TOTAL_EXPERIMENTS=$(wc -l < "$QUEUE_FILE")
else
    > "$QUEUE_FILE"
    for num_clients in "${CONCURRENCY_VALUES[@]}"; do
        for scheduler in $SCHEDULERS; do
            echo "${scheduler}|${num_clients}" >> "$QUEUE_FILE"
        done
    done
    TOTAL_EXPERIMENTS=$(wc -l < "$QUEUE_FILE")
fi

echo "总实验数: $TOTAL_EXPERIMENTS"
echo "  = ${#CONCURRENCY_VALUES[@]} 并发度 × $(echo $SCHEDULERS | wc -w) 调度器"
echo ""

# 保存全局配置
cat > "${OUTPUT_DIR}/experiment_config.json" << EOF
{
    "experiment_type": "concurrency_sweep",
    "purpose": "Sweep num_clients to evaluate TTFT/TPOT trade-off at moderate concurrency and find SLO-constrained max throughput",
    "dataset_path": "${DATASET_PATH}",
    "model": "${MODEL}",
    "gpu_type": "${GPU_TYPE}",
    "max_turns": ${MAX_TURNS},
    "limit_max_tokens": ${LIMIT_MAX_TOKENS},
    "request_timeout": ${REQUEST_TIMEOUT},
    "k_ratio": ${K_RATIO},
    "concurrency_values": [$(echo "${CONCURRENCY_VALUES[*]}" | sed 's/ /, /g')],
    "schedulers": [$(echo "$SCHEDULERS" | sed 's/[^ ]*/"&"/g' | sed 's/ /, /g')],
    "optimal_configs": {
$(for scheduler in $SCHEDULERS; do
    config=$(get_optimal_config "$GPU_TYPE" "$MODEL_SHORT" "$scheduler")
    read -r tb bs <<< "$config"
    echo "        \"${scheduler}\": {\"tb\": ${tb}, \"bs\": ${bs}},"
done | sed '$ s/,$//')
    },
    "calibration_file": "${VLLM_PD_CALIBRATION_FILE}",
    "gpus_used": [$(IFS=,; echo "${GPUS_TO_USE[*]}")],
    "total_experiments": ${TOTAL_EXPERIMENTS},
    "timestamp": "$(date -Iseconds)"
}
EOF

echo "配置已保存: ${OUTPUT_DIR}/experiment_config.json"
echo ""

# ========================================
# 运行单个实验
# ========================================
run_experiment() {
    local gpu_id=$1 scheduler=$2 num_clients=$3

    # 获取该 scheduler 的最优配置
    local config
    config=$(get_optimal_config "$GPU_TYPE" "$MODEL_SHORT" "$scheduler") || return 1
    read -r tb bs <<< "$config"

    local port=$((BASE_PORT + gpu_id))
    local result_dir="${OUTPUT_DIR}/clients_${num_clients}"
    local log_file="${result_dir}/logs/${scheduler}.log"
    local bench_log="${result_dir}/logs/${scheduler}_bench.log"
    local result_file="${result_dir}/bench_${scheduler}.json"

    # 检查是否跳过已有结果
    if [ "${SKIP_EXISTING:-1}" = "1" ] && [ -f "$result_file" ]; then
        echo "[GPU $gpu_id] 跳过: ${scheduler} clients=${num_clients} (结果已存在)"
        return 0
    fi

    mkdir -p "${result_dir}/logs"
    : > "$log_file"
    : > "$bench_log"

    check_port_available $port $gpu_id || return 1

    echo "[GPU $gpu_id] 开始: ${scheduler} clients=${num_clients} (TB=${tb}, BS=${bs})"

    # 设置环境变量
    export CUDA_VISIBLE_DEVICES=$gpu_id
    export VLLM_COLLECT_SCHEDULE_STATS=1

    case "$scheduler" in
        baseline)
            export VLLM_USE_PD_SCHEDULER=0
            unset VLLM_PD_K_MODE VLLM_PD_K_STAR VLLM_PD_K_RATIO
            ;;
        pd_ratio)
            export VLLM_USE_PD_SCHEDULER=1
            export VLLM_PD_K_MODE=ratio
            export VLLM_PD_K_RATIO=$K_RATIO
            unset VLLM_PD_K_STAR
            ;;
        pd_ifr)
            export VLLM_USE_PD_SCHEDULER=1
            export VLLM_PD_K_MODE=ifr
            unset VLLM_PD_K_RATIO VLLM_PD_K_STAR
            ;;
    esac

    wait_for_gpu_memory $gpu_id 60 || return 1

    # 启动服务
    local dtype_arg=""
    if [ -n "${DTYPE:-}" ]; then
        dtype_arg="--dtype $DTYPE"
    fi

    VLLM_SCHEDULE_STATS_FILE="${result_dir}/${scheduler}_stats.json" \
    vllm serve "$MODEL" \
        --port "$port" \
        --gpu-memory-utilization 0.9 \
        --max-num-seqs "$bs" \
        --max-num-batched-tokens "$tb" \
        $dtype_arg >> "$log_file" 2>&1 &
    local server_pid=$!

    if ! wait_for_server $port $server_pid 180 "$log_file"; then
        echo "[GPU $gpu_id] 服务启动失败: ${scheduler} clients=${num_clients}"
        kill_server $server_pid $gpu_id
        return 1
    fi

    # 运行多轮对话 benchmark
    local _bench_dir="${SCRIPT_DIR}/../../../benchmarks/multi_turn"
    ( cd "$_bench_dir" && python benchmark_serving_multi_turn_threaded.py \
        --input-file "$DATASET_PATH" \
        --model "$MODEL" \
        --url "http://localhost:${port}" \
        --num-clients "$num_clients" \
        --max-turns "$MAX_TURNS" \
        --limit-min-tokens -1 \
        --limit-max-tokens "$LIMIT_MAX_TOKENS" \
        --request-timeout-sec "$REQUEST_TIMEOUT" \
        --output-file "${result_dir}/${scheduler}_conversations.json" \
        --metrics-file "${result_dir}/bench_${scheduler}.json" ) \
        > "$bench_log" 2>&1
    local bench_status=$?

    kill_server $server_pid $gpu_id

    if [ $bench_status -eq 0 ]; then
        echo "[GPU $gpu_id] 完成: ${scheduler} clients=${num_clients}"
    else
        echo "[GPU $gpu_id] 失败: ${scheduler} clients=${num_clients}"
    fi

    return $bench_status
}

# ========================================
# 并行调度
# ========================================
PROGRESS_FILE="${OUTPUT_DIR}/progress.txt"
LOCK_FILE="${OUTPUT_DIR}/.queue.lock"

gpu_worker() {
    local gpu_id=$1

    while true; do
        local exp=$(get_next_experiment "$QUEUE_FILE" "$LOCK_FILE")
        [ -z "$exp" ] && break

        IFS='|' read -r scheduler num_clients <<< "$exp"

        if run_experiment "$gpu_id" "$scheduler" "$num_clients"; then
            update_progress "OK|${exp}" "$PROGRESS_FILE" "$LOCK_FILE" "$TOTAL_EXPERIMENTS"
        else
            update_progress "FAIL|${exp}" "$PROGRESS_FILE" "$LOCK_FILE" "$TOTAL_EXPERIMENTS"
        fi
    done
}

# ========================================
# 主流程
# ========================================
echo "开始并行执行..."
echo "========================================"

> "$PROGRESS_FILE"

for gpu_id in "${GPUS_TO_USE[@]}"; do
    gpu_worker "$gpu_id" &
    WORKER_PIDS+=($!)
    echo "启动 GPU $gpu_id worker (PID: ${WORKER_PIDS[-1]})"
    sleep 10
done

echo ""
echo "监控进度: watch -n 5 'wc -l ${PROGRESS_FILE}'"
echo ""

for pid in "${WORKER_PIDS[@]}"; do
    wait $pid || true
done

print_summary "$PROGRESS_FILE" "$TOTAL_EXPERIMENTS" "$OUTPUT_DIR"
echo ""
echo "========================================"
echo "实验完成!"
echo "========================================"
echo ""
echo "结果目录: $OUTPUT_DIR"
echo ""
echo "运行分析脚本:"
echo "  python pd_exp/plot_concurrency_latency.py $OUTPUT_DIR"
echo ""
echo "多模型运行示例:"
echo "  MODEL=Qwen/Qwen3-30B-A3B $0 $MAX_GPUS"
echo "  MODEL=google/gemma-3-1b-it $0 $MAX_GPUS"
