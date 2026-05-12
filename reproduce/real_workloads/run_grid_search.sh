#!/bin/bash

# TB × BS 网格搜索实验脚本 (真实数据集版本)
# 对比 baseline 和 PD scheduler 在所有 (TB, BS) 组合下的性能
#
# 用法: ./run_grid_search_real.sh <DATASET_PATH> [MAX_GPUS]
#
# 示例:
#   # 先导出数据集
#   python experiments/serve/export_dataset.py \
#       --dataset alpaca \
#       --model Qwen/Qwen3-8B \
#       --num-samples 4000 \
#       --output ./experiments/serve/alpaca_prompts.jsonl
#
#   # 运行实验
#   ./run_grid_search_real.sh ./experiments/serve/alpaca_prompts.jsonl 4

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common/common.sh"

WORKER_PIDS=()
cleanup() {
    for pid in "${WORKER_PIDS[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM HUP

# 检查参数
if [ -z "${1:-}" ]; then
    echo "用法: $0 <DATASET_PATH> [MAX_GPUS]"
    echo ""
    echo "必须提供数据集文件路径 (JSONL 格式)"
    echo ""
    echo "示例:"
    echo "  # 先导出数据集"
    echo "  python experiments/serve/export_dataset.py \\"
    echo "      --dataset alpaca \\"
    echo "      --model Qwen/Qwen3-8B \\"
    echo "      --num-samples 4000 \\"
    echo "      --output ./experiments/serve/alpaca_prompts.jsonl"
    echo ""
    echo "  # 运行实验"
    echo "  $0 ./experiments/serve/alpaca_prompts.jsonl 4"
    exit 1
fi

DATASET_PATH="$1"
MAX_GPUS=${2:-4}

# 检查数据集文件
if [ ! -f "$DATASET_PATH" ]; then
    echo "错误: 数据集文件不存在: $DATASET_PATH"
    exit 1
fi

if [[ "$DATASET_PATH" != *.jsonl ]]; then
    echo "错误: 数据集文件必须是 JSONL 格式 (.jsonl)"
    echo "请使用 export_dataset.py 导出数据集"
    exit 1
fi

# 获取数据集名称（用于输出目录）
DATASET_NAME=$(basename "$DATASET_PATH" .jsonl)

# 实验参数
MODEL=${MODEL:-"Qwen/Qwen3-8B"}
# 模型短名称（用于目录命名，将 / 替换为 _）
MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||')
NUM_PROMPTS=${NUM_PROMPTS:-4000}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-2048}
NUM_WARMUP_REQUESTS=${NUM_WARMUP_REQUESTS:-20}
K_RATIO=${K_RATIO:-0.8}
BASE_PORT=${BASE_PORT:-11000}
CUSTOM_OUTPUT_LEN=${CUSTOM_OUTPUT_LEN:-4000}
ENABLE_THINKING=${ENABLE_THINKING:-true}  # 控制 Qwen3 thinking 模式

# 硬件校准文件 (必须，按模型区分)
if [ -z "${VLLM_PD_CALIBRATION_FILE:-}" ]; then
    DEFAULT_CALIBRATION="${SCRIPT_DIR}/../outputs/pd_calibration_${MODEL_SHORT}.json"
    if [ -f "$DEFAULT_CALIBRATION" ]; then
        export VLLM_PD_CALIBRATION_FILE="$DEFAULT_CALIBRATION"
    else
        echo "错误: 未找到硬件校准文件!"
        echo ""
        echo "PD Scheduler 需要硬件校准参数才能准确调度。"
        echo "请先运行校准:"
        echo "  python -m vllm.v1.core.sched.calibration --model ${MODEL} --output ${DEFAULT_CALIBRATION}"
        echo ""
        echo "校准文件默认保存到: ${DEFAULT_CALIBRATION}"
        echo "或手动指定: VLLM_PD_CALIBRATION_FILE=/path/to/file.json $0 ..."
        exit 1
    fi
fi
echo "使用校准文件: $VLLM_PD_CALIBRATION_FILE"

# 从校准文件中读取 alpha/beta 参数
if [ -f "$VLLM_PD_CALIBRATION_FILE" ]; then
    ALPHA_P=$(python3 -c "import json; print(json.load(open('$VLLM_PD_CALIBRATION_FILE'))['alpha_p'])" 2>/dev/null || echo "null")
    BETA_P=$(python3 -c "import json; print(json.load(open('$VLLM_PD_CALIBRATION_FILE'))['beta_p'])" 2>/dev/null || echo "null")
    ALPHA_D=$(python3 -c "import json; print(json.load(open('$VLLM_PD_CALIBRATION_FILE'))['alpha_d'])" 2>/dev/null || echo "null")
    BETA_D=$(python3 -c "import json; print(json.load(open('$VLLM_PD_CALIBRATION_FILE'))['beta_d'])" 2>/dev/null || echo "null")
    echo "  alpha_p: $ALPHA_P, beta_p: $BETA_P"
    echo "  alpha_d: $ALPHA_D, beta_d: $BETA_D"
else
    ALPHA_P="null"
    BETA_P="null"
    ALPHA_D="null"
    BETA_D="null"
fi

# 网格搜索参数 (不需要 SCENARIOS，因为真实数据集的分布是固定的)
BS_VALUES=(256 512 1024 1536 2048)
TB_VALUES=(4096 8192 10240 14336 16384 18432)

# 输出目录 (包含模型名)
OUTPUT_DIR="${SCRIPT_DIR}/../outputs/grid_search_${DATASET_NAME}_${MODEL_SHORT}_Con_${MAX_CONCURRENCY}_Prompts_${NUM_PROMPTS}"
mkdir -p "$OUTPUT_DIR"

# 初始化环境
init_experiment_env

echo "========================================"
echo "TB × BS 网格搜索实验 (真实数据集)"
echo "========================================"

# 检测并选择 GPU
select_gpus $MAX_GPUS

echo ""
echo "实验配置:"
echo "  DATASET: $DATASET_PATH"
echo "  MODEL: $MODEL"
echo "  DTYPE: ${DTYPE:-auto}"
echo "  NUM_PROMPTS: $NUM_PROMPTS"
echo "  MAX_CONCURRENCY: $MAX_CONCURRENCY"
echo "  CUSTOM_OUTPUT_LEN: $CUSTOM_OUTPUT_LEN"
echo "  ENABLE_THINKING: $ENABLE_THINKING"
echo "  K_RATIO (for pd_ratio): $K_RATIO"
echo "  BS_VALUES: ${BS_VALUES[*]}"
echo "  TB_VALUES: ${TB_VALUES[*]}"
# 支持通过 SCHEDULERS 环境变量指定要运行的调度器
# 例如: SCHEDULERS="pd_ifr" 只运行 pd_ifr 模式
SCHEDULERS=${SCHEDULERS:-"baseline pd_ratio pd_ifr"}
echo "  SCHEDULERS: $SCHEDULERS"
# 支持版本后缀，用于重复运行同一调度器生成不同结果文件
# 例如: VERSION=1 SCHEDULERS="pd_ifr" 会生成 bench_pd_ifr_1.json
if [ -n "${VERSION:-}" ]; then
    echo "  VERSION: ${VERSION} (文件后缀: _${VERSION})"
fi
echo "  CALIBRATION_FILE: ${VLLM_PD_CALIBRATION_FILE:-"(未设置，使用默认参数)"}"
echo ""

# 生成实验队列
QUEUE_FILE="${OUTPUT_DIR}/experiment_queue.txt"
RESUME=${RESUME:-false}

if [ "$RESUME" = "true" ] && [ -f "$QUEUE_FILE" ] && [ -s "$QUEUE_FILE" ]; then
    echo "恢复模式: 使用现有队列文件 ($QUEUE_FILE)"
    TOTAL_EXPERIMENTS=$(wc -l < "$QUEUE_FILE")
else
    > "$QUEUE_FILE"
    for tb in "${TB_VALUES[@]}"; do
        for bs in "${BS_VALUES[@]}"; do
            for scheduler in $SCHEDULERS; do
                echo "${scheduler}|${bs}|${tb}" >> "$QUEUE_FILE"
            done
        done
    done
    TOTAL_EXPERIMENTS=$(wc -l < "$QUEUE_FILE")
fi
echo "总实验数: $TOTAL_EXPERIMENTS"
echo ""

# 保存全局配置
cat > "${OUTPUT_DIR}/experiment_config.json" << EOF
{
    "dataset_path": "${DATASET_PATH}",
    "dataset_name": "${DATASET_NAME}",
    "model": "${MODEL}",
    "dtype": "${DTYPE:-auto}",
    "num_prompts": ${NUM_PROMPTS},
    "max_concurrency": ${MAX_CONCURRENCY},
    "custom_output_len": ${CUSTOM_OUTPUT_LEN},
    "enable_thinking": ${ENABLE_THINKING},
    "k_ratio": ${K_RATIO},
    "bs_values": [$(echo "${BS_VALUES[*]}" | sed 's/ /, /g')],
    "tb_values": [$(echo "${TB_VALUES[*]}" | sed 's/ /, /g')],
    "schedulers": [$(echo "$SCHEDULERS" | sed 's/[^ ]*/"&"/g' | sed 's/ /, /g')],
    "scheduler_descriptions": {
        "baseline": "vLLM default scheduler",
        "pd_ratio": "PD scheduler with ratio mode (θ*=${K_RATIO})",
        "pd_ifr": "PD scheduler with IFR mode (adaptive θ* based on hazard rate)"
    },
    "calibration_file": "${VLLM_PD_CALIBRATION_FILE:-null}",
    "calibration_params": {
        "alpha_p": ${ALPHA_P},
        "beta_p": ${BETA_P},
        "alpha_d": ${ALPHA_D},
        "beta_d": ${BETA_D}
    },
    "gpus_used": [$(IFS=,; echo "${GPUS_TO_USE[*]}")],
    "total_experiments": ${TOTAL_EXPERIMENTS},
    "timestamp": "$(date -Iseconds)"
}
EOF

# 运行单个实验
run_experiment() {
    local gpu_id=$1 scheduler=$2 bs=$3 tb=$4
    local port=$((BASE_PORT + gpu_id))
    local result_dir="${OUTPUT_DIR}/tb${tb}/bs${bs}"
    # 支持版本后缀: VERSION=1 会生成 pd_ifr_1.log, bench_pd_ifr_1.json 等
    local suffix=""
    if [ -n "${VERSION:-}" ]; then
        suffix="_${VERSION}"
    fi
    local log_file="${result_dir}/logs/${scheduler}${suffix}.log"
    local result_file="${result_dir}/bench_${scheduler}${suffix}.json"

    # 检查是否跳过已有结果
    if [ "${SKIP_EXISTING:-1}" = "1" ] && [ -f "$result_file" ]; then
        echo "[GPU $gpu_id] 跳过: ${scheduler} tb=${tb} bs=${bs} (结果已存在)"
        return 0
    fi

    mkdir -p "${result_dir}/logs"
    : > "$log_file"

    check_port_available $port $gpu_id || return 1

    echo "[GPU $gpu_id] 开始: ${scheduler} tb=${tb} bs=${bs}"

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

    VLLM_SCHEDULE_STATS_FILE="${result_dir}/${scheduler}${suffix}_stats.json" \
    vllm serve "$MODEL" \
        --port "$port" \
        --gpu-memory-utilization 0.9 \
        --max-num-seqs "$bs" \
        --max-num-batched-tokens "$tb" \
        $dtype_arg >> "$log_file" 2>&1 &
    local server_pid=$!

    if ! wait_for_server $port $server_pid 180 "$log_file"; then
        echo "[GPU $gpu_id] 服务启动失败"
        kill_server $server_pid $gpu_id
        return 1
    fi

    # 构建 benchmark 命令
    local bench_cmd=(
        vllm bench serve
        --model "$MODEL"
        --base-url "http://localhost:${port}"
        --dataset-name custom
        --dataset-path "$DATASET_PATH"
        --custom-output-len "$CUSTOM_OUTPUT_LEN"
        --num-prompts "$NUM_PROMPTS"
        --num-warmups "$NUM_WARMUP_REQUESTS"
        --request-rate inf
        --max-concurrency "$MAX_CONCURRENCY"
        --save-result
        --save-detailed
        --result-dir "${result_dir}"
        --result-filename "bench_${scheduler}${suffix}.json"
    )

    # 如果关闭 thinking 模式，需要使用 chat backend 并添加 extra-body 参数
    # --backend openai-chat 会正确将 prompt 包装为 messages 格式
    # --endpoint 必须同时指定为 /v1/chat/completions
    if [ "$ENABLE_THINKING" = "false" ]; then
        bench_cmd+=(--backend openai-chat)
        bench_cmd+=(--endpoint /v1/chat/completions)
        bench_cmd+=(--extra-body '{"chat_template_kwargs":{"enable_thinking":false}}')
    fi

    # 运行 benchmark
    "${bench_cmd[@]}" >> "$log_file" 2>&1
    local bench_status=$?

    kill_server $server_pid $gpu_id

    if [ $bench_status -eq 0 ]; then
        echo "[GPU $gpu_id] 完成: ${scheduler} tb=${tb} bs=${bs}"
    else
        echo "[GPU $gpu_id] 失败: ${scheduler} tb=${tb} bs=${bs}"
    fi

    return $bench_status
}

# 并行调度
PROGRESS_FILE="${OUTPUT_DIR}/progress.txt"
LOCK_FILE="${OUTPUT_DIR}/.queue.lock"

gpu_worker() {
    local gpu_id=$1

    while true; do
        local exp=$(get_next_experiment "$QUEUE_FILE" "$LOCK_FILE")
        [ -z "$exp" ] && break

        IFS='|' read -r scheduler bs tb <<< "$exp"

        if run_experiment "$gpu_id" "$scheduler" "$bs" "$tb"; then
            update_progress "OK|${exp}" "$PROGRESS_FILE" "$LOCK_FILE" "$TOTAL_EXPERIMENTS"
        else
            update_progress "FAIL|${exp}" "$PROGRESS_FILE" "$LOCK_FILE" "$TOTAL_EXPERIMENTS"
        fi
    done
}

# 主流程
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
echo "运行分析脚本:"
echo "  # Grid search 结果分析 (真实数据集)"
echo "  python ${SCRIPT_DIR}/analyze_grid_search.py $OUTPUT_DIR"
echo ""
echo "  # Input/Output 长度统计 (查看是否 decode-heavy)"
echo "  python ${SCRIPT_DIR}/../analyze_benchmark_stats.py $OUTPUT_DIR --summary-only"
