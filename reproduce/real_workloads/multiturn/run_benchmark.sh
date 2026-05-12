#!/bin/bash

# 多轮对话 Benchmark 脚本 (Prefix Cache 测试)
# 对比 baseline 和 PD scheduler 在多轮对话场景下的性能
#
# 用法: ./run_benchmark.sh <DATASET_PATH> [MAX_GPUS]
#
# 示例:
#   # 先导出数据集
#   python pd_exp/multiturn/export_dataset.py \
#       --dataset wildchat \
#       --model Qwen/Qwen3-8B \
#       --num-conversations 500 \
#       --min-turns 8 \
#       --output ./outputs/wildchat_multiturn.json
#
#   # 运行实验
#   ./reproduce/real_workloads/multiturn/run_benchmark.sh ./outputs/wildchat_multiturn.json 4

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

# 检查参数
if [ -z "${1:-}" ]; then
    echo "用法: $0 <DATASET_PATH> [MAX_GPUS]"
    echo ""
    echo "必须提供数据集文件路径 (JSON 格式，多轮对话)"
    echo ""
    echo "示例:"
    echo "  # 先导出数据集"
    echo "  python pd_exp/multiturn/export_dataset.py \\"
    echo "      --dataset wildchat \\"
    echo "      --model Qwen/Qwen3-8B \\"
    echo "      --num-conversations 500 \\"
    echo "      --min-turns 8 \\"
    echo "      --output ./outputs/wildchat_multiturn.json"
    echo ""
    echo "  # 运行实验"
    echo "  $0 ./outputs/wildchat_multiturn.json 4"
    exit 1
fi

DATASET_PATH="$1"
MAX_GPUS=${2:-4}

# 检查数据集文件
if [ ! -f "$DATASET_PATH" ]; then
    echo "错误: 数据集文件不存在: $DATASET_PATH"
    exit 1
fi

if [[ "$DATASET_PATH" != *.json ]]; then
    echo "错误: 数据集文件必须是 JSON 格式 (.json)"
    echo "请使用 pd_exp/multiturn/export_dataset.py 导出数据集"
    exit 1
fi

# 获取数据集名称（用于输出目录）
DATASET_NAME=$(basename "$DATASET_PATH" .json)

# 实验参数
MODEL=${MODEL:-"Qwen/Qwen3-8B"}
# 模型短名称（用于目录命名，将 / 替换为 _）
MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||')
NUM_CLIENTS=${NUM_CLIENTS:-2048}
MAX_TURNS=${MAX_TURNS:-12}
LIMIT_MAX_TOKENS=${LIMIT_MAX_TOKENS:-256}
REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-120}
BASE_PORT=${BASE_PORT:-10000}
K_RATIO=${K_RATIO:-0.8}

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

# 网格搜索参数
BS_VALUES=(${BS_VALUES:-256 512 1024 1536 2048})
TB_VALUES=(${TB_VALUES:-4096 8192 10240 14336 16384 18432})

# 输出目录 (包含模型名)
OUTPUT_DIR="${SCRIPT_DIR}/../outputs/multiturn_${DATASET_NAME}_${MODEL_SHORT}_Clients_${NUM_CLIENTS}_MaxTurns_${MAX_TURNS}"
mkdir -p "$OUTPUT_DIR"

# 初始化环境
init_experiment_env

echo "========================================"
echo "多轮对话 Benchmark (Prefix Cache 测试)"
echo "========================================"

# 检测并选择 GPU
select_gpus $MAX_GPUS

echo ""
echo "实验配置:"
echo "  DATASET: $DATASET_PATH"
echo "  MODEL: $MODEL"
echo "  DTYPE: ${DTYPE:-auto}"
echo "  NUM_CLIENTS: $NUM_CLIENTS"
echo "  MAX_TURNS: $MAX_TURNS"
echo "  LIMIT_MAX_TOKENS: $LIMIT_MAX_TOKENS"
echo "  K_RATIO (for pd_ratio): $K_RATIO"
echo "  BS_VALUES: ${BS_VALUES[*]}"
echo "  TB_VALUES: ${TB_VALUES[*]}"
# 支持通过 SCHEDULERS 环境变量指定要运行的调度器
SCHEDULERS=${SCHEDULERS:-"baseline pd_ratio pd_ifr"}
echo "  SCHEDULERS: $SCHEDULERS"
echo "  CALIBRATION_FILE: ${VLLM_PD_CALIBRATION_FILE:-"(未设置)"}"
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
    "num_clients": ${NUM_CLIENTS},
    "max_turns": ${MAX_TURNS},
    "limit_max_tokens": ${LIMIT_MAX_TOKENS},
    "request_timeout": ${REQUEST_TIMEOUT},
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

# Python 脚本: 从 benchmark 输出中提取并保存 metrics
extract_metrics_script() {
    cat << 'PYTHON_SCRIPT'
import sys
import json
import re
from pathlib import Path

def extract_metrics(bench_log_path, output_path, duration_sec):
    """从 benchmark 日志中提取 metrics 并保存为 JSON"""
    metrics = {}

    with open(bench_log_path, 'r') as f:
        content = f.read()

    # 解析 pandas describe() 输出
    # 格式: metric_name  count  mean  std  min  25%  50%  75%  90%  99%  max
    lines = content.strip().split('\n')
    for line in lines:
        parts = line.split()
        if len(parts) >= 3:
            metric_name = parts[0]
            if metric_name in ['ttft_ms', 'tpot_ms', 'latency_ms',
                               'input_num_tokens', 'output_num_tokens',
                               'input_num_turns', 'output_num_chunks']:
                try:
                    # count=1, mean=2, std=3, min=4, 25%=5, 50%=6, 75%=7, 90%=8, 99%=9, max=10
                    count = float(parts[1])
                    mean = float(parts[2])
                    std = float(parts[3]) if len(parts) > 3 else 0
                    min_val = float(parts[4]) if len(parts) > 4 else mean
                    p50 = float(parts[6]) if len(parts) > 6 else mean
                    p99 = float(parts[9]) if len(parts) > 9 else mean
                    max_val = float(parts[10]) if len(parts) > 10 else mean

                    metrics[f'mean_{metric_name}'] = mean
                    metrics[f'std_{metric_name}'] = std
                    metrics[f'min_{metric_name}'] = min_val
                    metrics[f'median_{metric_name}'] = p50
                    metrics[f'p99_{metric_name}'] = p99
                    metrics[f'max_{metric_name}'] = max_val
                    metrics[f'count_{metric_name}'] = count
                except (ValueError, IndexError):
                    pass

    # 计算 throughput
    if 'count_latency_ms' in metrics and duration_sec > 0:
        completed = metrics['count_latency_ms']
        metrics['request_throughput'] = completed / duration_sec

        if 'mean_output_num_tokens' in metrics:
            total_output_tokens = completed * metrics['mean_output_num_tokens']
            metrics['output_throughput'] = total_output_tokens / duration_sec

    # 重命名以匹配 vllm bench serve 格式
    rename_map = {
        'mean_ttft_ms': 'mean_ttft_ms',
        'median_ttft_ms': 'median_ttft_ms',
        'p99_ttft_ms': 'p99_ttft_ms',
        'mean_tpot_ms': 'mean_tpot_ms',
        'median_tpot_ms': 'median_tpot_ms',
        'p99_tpot_ms': 'p99_tpot_ms',
        'mean_latency_ms': 'mean_e2e_latency_ms',
        'median_latency_ms': 'median_e2e_latency_ms',
        'p99_latency_ms': 'p99_e2e_latency_ms',
    }

    result = {}
    for old_key, new_key in rename_map.items():
        if old_key in metrics:
            result[new_key] = metrics[old_key]

    # 添加 throughput
    if 'request_throughput' in metrics:
        result['request_throughput'] = metrics['request_throughput']
    if 'output_throughput' in metrics:
        result['output_throughput'] = metrics['output_throughput']

    # 添加其他有用的 metrics
    for key in ['mean_input_num_tokens', 'mean_output_num_tokens',
                'mean_input_num_turns', 'count_latency_ms']:
        if key in metrics:
            result[key] = metrics[key]

    # 保存
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    return result

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: extract_metrics.py <bench_log> <output_json> <duration_sec>")
        sys.exit(1)

    bench_log = sys.argv[1]
    output_json = sys.argv[2]
    duration_sec = float(sys.argv[3])

    result = extract_metrics(bench_log, output_json, duration_sec)
    print(f"Saved metrics to {output_json}: throughput={result.get('request_throughput', 0):.2f} req/s")
PYTHON_SCRIPT
}

# 运行单个实验
run_experiment() {
    local gpu_id=$1 scheduler=$2 bs=$3 tb=$4
    local port=$((BASE_PORT + gpu_id))
    local result_dir="${OUTPUT_DIR}/tb${tb}/bs${bs}"
    local log_file="${result_dir}/logs/${scheduler}.log"
    local bench_log="${result_dir}/logs/${scheduler}_bench.log"
    local result_file="${result_dir}/bench_${scheduler}.json"

    # 检查是否跳过已有结果
    if [ "${SKIP_EXISTING:-1}" = "1" ] && [ -f "$result_file" ]; then
        echo "[GPU $gpu_id] 跳过: ${scheduler} tb=${tb} bs=${bs} (结果已存在)"
        return 0
    fi

    mkdir -p "${result_dir}/logs"
    : > "$log_file"
    : > "$bench_log"

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

    VLLM_SCHEDULE_STATS_FILE="${result_dir}/${scheduler}_stats.json" \
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

    # 运行多轮对话 benchmark
    python benchmarks/multi_turn/benchmark_serving_multi_turn_threaded.py \
        --input-file "$DATASET_PATH" \
        --model "$MODEL" \
        --url "http://localhost:${port}" \
        --num-clients "$NUM_CLIENTS" \
        --max-turns "$MAX_TURNS" \
        --limit-min-tokens -1 \
        --limit-max-tokens "$LIMIT_MAX_TOKENS" \
        --request-timeout-sec "$REQUEST_TIMEOUT" \
        --output-file "${result_dir}/${scheduler}_conversations.json" \
        --metrics-file "${result_dir}/bench_${scheduler}.json" \
        > "$bench_log" 2>&1
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
echo "========================================"
echo "实验完成!"
echo "========================================"
echo ""
echo "结果目录: $OUTPUT_DIR"
echo ""
echo "运行分析脚本:"
echo "  # 结果分析 (scheduler 对比)"
echo "  python ${SCRIPT_DIR}/analyze_results.py $OUTPUT_DIR"
