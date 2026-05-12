#!/bin/bash
#
# Hazard Rate Ordering 实验：验证 k*_DFR < k*_CFR < k*_IFR
#
# 验证内容：
#   使用 Gamma 分布验证不同 hazard rate 类型下的 k* 排序
#   - shape < 1: DFR (Decreasing Failure Rate)
#   - shape = 1: CFR (Constant Failure Rate, Exponential)
#   - shape > 1: IFR (Increasing Failure Rate)
#
# 参数设置：
#   a ∈ {0.5, 1, 2} (shape)
#   b ∈ {256, 128, 64} (scale)
#   E[O] = a × b = 128 (保持均值一致)
#   N = 256
#
# 用法：
#   ./run_hazard_rate_experiment.sh 4          # 使用 4 个 GPU
#   SKIP_EXISTING=1 ./run_hazard_rate_experiment.sh 4  # 跳过已有结果
# 分析数据: python reproduce/hazard_rate/analyze_hazard_rate.py outputs/hazard_rate_ordering_N256_O128
# ==================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common/common.sh"

# 清理函数
WORKER_PIDS=()
cleanup() {
    for pid in "${WORKER_PIDS[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM HUP

# ========================================
# 命令行参数与环境变量配置
# ========================================
MAX_GPUS=${1:-4}
MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-256}        # Baseline 使用
PD_MAX_BATCH_SIZE=${PD_MAX_BATCH_SIZE:-256}  # P/D 分离使用 (N=256)
NUM_PROMPTS=${NUM_PROMPTS:-3000}
MODEL=${MODEL:-"Qwen/Qwen3-8B"}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-3000}
RANDOM_RANGE_RATIO=${RANDOM_RANGE_RATIO:-0}
BASE_PORT=${BASE_PORT:-10000}

# K_STAR 值列表：均匀采样 N 范围内的 k 值
K_STAR_VALUES=()
NUM_K_SAMPLES=${NUM_K_SAMPLES:-30}
if [ -n "$PD_MAX_BATCH_SIZE" ]; then
    for ((i=1; i<=NUM_K_SAMPLES; i++)); do
        k=$((PD_MAX_BATCH_SIZE * i / NUM_K_SAMPLES))
        K_STAR_VALUES+=($k)
    done
fi

# 测试类型开关
RUN_BASELINE=${RUN_BASELINE:-1}
RUN_KSTAR=${RUN_KSTAR:-1}   # 默认开启 k* 扫描
RUN_DIRECT=${RUN_DIRECT:-0}
SKIP_EXISTING=${SKIP_EXISTING:-0}

# 重复运行次数 (用于计算置信区间)
NUM_REPEATS=${NUM_REPEATS:-3}

# Warmup 配置
NUM_WARMUP_REQUESTS=${NUM_WARMUP_REQUESTS:-100}

# Token Budget 参数
TOKEN_BUDGET_DEFAULT=${TOKEN_BUDGET_DEFAULT:-}
TOKEN_BUDGET_PD=${TOKEN_BUDGET_PD:-16384}

# P/D Scheduler max_num_seqs
PD_MAX_NUM_SEQS=${PD_MAX_NUM_SEQS:-256}

# N 计算模式
PD_N_MODE=${PD_N_MODE:-reactive}
PD_OOM_TOLERANCE=${PD_OOM_TOLERANCE:-0.01}

# 硬件校准文件
if [ -z "${VLLM_PD_CALIBRATION_FILE:-}" ]; then
    DEFAULT_CALIBRATION="${SCRIPT_DIR}/../outputs/pd_calibration.json"
    if [ -f "$DEFAULT_CALIBRATION" ]; then
        export VLLM_PD_CALIBRATION_FILE="$DEFAULT_CALIBRATION"
    else
        echo "错误: 未找到硬件校准文件!"
        echo "请先运行校准: python -m vllm.v1.core.sched.calibration --model ${MODEL}"
        exit 1
    fi
fi
echo "使用校准文件: $VLLM_PD_CALIBRATION_FILE"

# 读取校准参数
if [ -f "$VLLM_PD_CALIBRATION_FILE" ]; then
    ALPHA_P=$(python3 -c "import json; print(json.load(open('$VLLM_PD_CALIBRATION_FILE'))['alpha_p'])" 2>/dev/null || echo "null")
    BETA_P=$(python3 -c "import json; print(json.load(open('$VLLM_PD_CALIBRATION_FILE'))['beta_p'])" 2>/dev/null || echo "null")
    ALPHA_D=$(python3 -c "import json; print(json.load(open('$VLLM_PD_CALIBRATION_FILE'))['alpha_d'])" 2>/dev/null || echo "null")
    BETA_D=$(python3 -c "import json; print(json.load(open('$VLLM_PD_CALIBRATION_FILE'))['beta_d'])" 2>/dev/null || echo "null")
    echo "  alpha_p: $ALPHA_P, beta_p: $BETA_P"
    echo "  alpha_d: $ALPHA_D, beta_d: $BETA_D"
fi

# ========================================
# Hazard Rate Ordering 实验配置
# ========================================
# Gamma 分布参数：E[O] = shape × scale = 128
# shape=0.5, scale=256 -> DFR
# shape=1.0, scale=128 -> CFR (Exponential)
# shape=2.0, scale=64  -> IFR
INPUT_LEN=1
OUTPUT_LEN=128

# Gamma shape/scale 配置 (format: "shape scale hazard_type")
GAMMA_CONFIGS=("0.5 256 DFR" "1.0 128 CFR" "2.0 64 IFR")

# 输出目录
OUTPUT_DIR="${OUTPUT_DIR:-"${SCRIPT_DIR}/../outputs/hazard_rate_ordering_N${PD_MAX_BATCH_SIZE}_O${OUTPUT_LEN}"}"
mkdir -p "$OUTPUT_DIR"

# 初始化环境
init_experiment_env

echo "========================================"
echo "Hazard Rate Ordering 实验"
echo "========================================"
echo "验证: k*_DFR < k*_CFR < k*_IFR"
echo ""
echo "Gamma 分布配置 (E[O] = ${OUTPUT_LEN}):"
for config in "${GAMMA_CONFIGS[@]}"; do
    read -r shape scale hazard_type <<< "$config"
    echo "  ${hazard_type}: shape=${shape}, scale=${scale}"
done

# 检测并选择 GPU
select_gpus $MAX_GPUS

echo ""
echo "实验配置:"
echo "  MODEL: $MODEL"
echo "  NUM_PROMPTS: $NUM_PROMPTS"
echo "  MAX_CONCURRENCY: $MAX_CONCURRENCY"
echo "  PD_MAX_BATCH_SIZE (N): $PD_MAX_BATCH_SIZE"
echo "  OUTPUT_LEN (E[O]): $OUTPUT_LEN"
echo "  INPUT_LEN (μ_L): $INPUT_LEN"
echo "  RUN_BASELINE: $RUN_BASELINE"
echo "  RUN_KSTAR: $RUN_KSTAR"
echo "  K_STAR_VALUES: ${K_STAR_VALUES[*]}"
echo "  NUM_REPEATS: $NUM_REPEATS"
echo ""

# ========================================
# 生成实验队列
# ========================================
QUEUE_FILE="${OUTPUT_DIR}/experiment_queue.txt"
> "$QUEUE_FILE"

for config in "${GAMMA_CONFIGS[@]}"; do
    read -r shape scale hazard_type <<< "$config"

    # Baseline 实验
    if [ "$RUN_BASELINE" = "1" ]; then
        for ((run=1; run<=NUM_REPEATS; run++)); do
            echo "baseline|${INPUT_LEN}|${OUTPUT_LEN}|${shape}|${hazard_type}||${run}" >> "$QUEUE_FILE"
        done
    fi

    # 固定 K* 扫描
    if [ "$RUN_KSTAR" = "1" ] && [ ${#K_STAR_VALUES[@]} -gt 0 ]; then
        for k_star in "${K_STAR_VALUES[@]}"; do
            for ((run=1; run<=NUM_REPEATS; run++)); do
                echo "kstar|${INPUT_LEN}|${OUTPUT_LEN}|${shape}|${hazard_type}|${k_star}|${run}" >> "$QUEUE_FILE"
            done
        done
    fi

    # Direct 模式
    if [ "$RUN_DIRECT" = "1" ]; then
        for ((run=1; run<=NUM_REPEATS; run++)); do
            echo "direct|${INPUT_LEN}|${OUTPUT_LEN}|${shape}|${hazard_type}||${run}" >> "$QUEUE_FILE"
        done
    fi
done

TOTAL_EXPERIMENTS=$(wc -l < "$QUEUE_FILE")
echo "总实验数: $TOTAL_EXPERIMENTS"
echo ""

# 保存实验配置
cat > "${OUTPUT_DIR}/experiment_config.json" << EOF
{
    "experiment": "hazard_rate_ordering",
    "description": "Verify k*_DFR < k*_CFR < k*_IFR for Gamma distributions with same mean",
    "model": "${MODEL}",
    "num_prompts": ${NUM_PROMPTS},
    "num_warmup_requests": ${NUM_WARMUP_REQUESTS},
    "max_concurrency": ${MAX_CONCURRENCY},
    "random_range_ratio": ${RANDOM_RANGE_RATIO},
    "output_distribution": "gamma",
    "fixed_params": {
        "N": ${PD_MAX_BATCH_SIZE},
        "E_O": ${OUTPUT_LEN},
        "mu_L": ${INPUT_LEN}
    },
    "gamma_configs": [
        {"shape": 0.5, "scale": 256, "type": "DFR", "mean": 128},
        {"shape": 1.0, "scale": 128, "type": "CFR", "mean": 128},
        {"shape": 2.0, "scale": 64, "type": "IFR", "mean": 128}
    ],
    "sweep_params": {
        "k_star": [$(IFS=,; echo "${K_STAR_VALUES[*]}")]
    },
    "max_batch_size_baseline": "${MAX_BATCH_SIZE:-null}",
    "pd_max_batch_size": "${PD_MAX_BATCH_SIZE:-null}",
    "pd_max_num_seqs": ${PD_MAX_NUM_SEQS},
    "token_budget_pd": ${TOKEN_BUDGET_PD},
    "run_baseline": ${RUN_BASELINE},
    "run_kstar": ${RUN_KSTAR},
    "run_direct": ${RUN_DIRECT},
    "num_repeats": ${NUM_REPEATS},
    "pd_n_mode": "${PD_N_MODE}",
    "calibration_file": "${VLLM_PD_CALIBRATION_FILE:-null}",
    "gpus_used": [$(IFS=,; echo "${GPUS_TO_USE[*]}")],
    "total_experiments": ${TOTAL_EXPERIMENTS},
    "timestamp": "$(date -Iseconds)"
}
EOF

# ========================================
# 运行单个实验
# ========================================
run_experiment() {
    local gpu_id=$1
    local exp_type=$2
    local input_len=$3
    local output_len=$4
    local gamma_shape=$5
    local hazard_type=$6
    local param=$7
    local run_num=$8

    local port=$((BASE_PORT + gpu_id))
    local scenario_name="${hazard_type}_shape${gamma_shape}"
    local result_dir="${OUTPUT_DIR}/${scenario_name}"
    local log_file result_prefix

    mkdir -p "${result_dir}/logs"

    # 根据实验类型设置参数
    case "$exp_type" in
        baseline)
            result_prefix="baseline"
            ;;
        kstar)
            result_prefix="fixed${param}"
            ;;
        direct)
            result_prefix="direct"
            ;;
    esac

    # 添加运行编号后缀
    if [ "$NUM_REPEATS" -gt 1 ] && [ -n "$run_num" ]; then
        result_prefix="${result_prefix}_run${run_num}"
    fi

    log_file="${result_dir}/logs/${result_prefix}.log"
    : > "$log_file"

    # 检查是否跳过已有结果
    if [ "$SKIP_EXISTING" = "1" ] && [ -f "${result_dir}/bench_${result_prefix}.json" ]; then
        echo "[GPU $gpu_id] 跳过: ${exp_type} ${scenario_name} run${run_num} (结果已存在)"
        return 0
    fi

    check_port_available $port $gpu_id || return 1

    local run_info=""
    if [ "$NUM_REPEATS" -gt 1 ] && [ -n "$run_num" ]; then
        run_info=" run${run_num}/${NUM_REPEATS}"
    fi
    echo "[GPU $gpu_id] 开始: ${exp_type} ${scenario_name} ${param:+k*=$param}${run_info}"

    # 设置环境变量
    export CUDA_VISIBLE_DEVICES=$gpu_id
    export VLLM_COLLECT_SCHEDULE_STATS=1

    # 构建 vllm serve 参数
    local serve_args="--gpu-memory-utilization 0.9"

    case "$exp_type" in
        baseline)
            export VLLM_USE_PD_SCHEDULER=0
            unset VLLM_PD_K_MODE VLLM_PD_K_STAR VLLM_PD_K_RATIO VLLM_PD_N_MODE VLLM_PD_OOM_TOLERANCE
            [ -n "$MAX_BATCH_SIZE" ] && serve_args="$serve_args --max-num-seqs $MAX_BATCH_SIZE"
            [ -n "$TOKEN_BUDGET_DEFAULT" ] && serve_args="$serve_args --max-num-batched-tokens $TOKEN_BUDGET_DEFAULT"
            ;;
        kstar)
            export VLLM_USE_PD_SCHEDULER=1
            export VLLM_PD_K_MODE=direct
            export VLLM_PD_K_STAR=$param
            export VLLM_PD_N_MODE=$PD_N_MODE
            export VLLM_PD_OOM_TOLERANCE=$PD_OOM_TOLERANCE
            unset VLLM_PD_K_RATIO
            serve_args="$serve_args --max-num-seqs $PD_MAX_BATCH_SIZE --max-num-batched-tokens $TOKEN_BUDGET_PD"
            ;;
        direct)
            export VLLM_USE_PD_SCHEDULER=1
            export VLLM_PD_K_MODE=direct
            export VLLM_PD_N_MODE=$PD_N_MODE
            export VLLM_PD_OOM_TOLERANCE=$PD_OOM_TOLERANCE
            unset VLLM_PD_K_RATIO VLLM_PD_K_STAR
            serve_args="$serve_args --max-num-seqs $PD_MAX_NUM_SEQS --max-num-batched-tokens $TOKEN_BUDGET_PD"
            ;;
    esac

    # 等待 GPU 内存可用
    wait_for_gpu_memory $gpu_id 60 || return 1

    # 启动服务
    VLLM_SCHEDULE_STATS_FILE="${result_dir}/${result_prefix}_stats.json" \
    vllm serve "$MODEL" \
        --port "$port" \
        $serve_args >> "$log_file" 2>&1 &
    local server_pid=$!

    # 等待服务启动
    if ! wait_for_server $port $server_pid 180 "$log_file"; then
        echo "[GPU $gpu_id] 服务启动失败: ${exp_type} ${scenario_name}"
        kill_server $server_pid $gpu_id
        return 1
    fi

    # 运行 benchmark - 使用 gamma_random 数据集
    vllm bench serve \
        --model "$MODEL" \
        --base-url "http://localhost:${port}" \
        --dataset-name gamma_random \
        --random-input-len "$input_len" \
        --random-output-len "$output_len" \
        --gamma-shape "$gamma_shape" \
        --random-range-ratio "$RANDOM_RANGE_RATIO" \
        --num-prompts "$NUM_PROMPTS" \
        --num-warmups "$NUM_WARMUP_REQUESTS" \
        --request-rate inf \
        --max-concurrency "$MAX_CONCURRENCY" \
        --save-result \
        --result-dir "${result_dir}" \
        --result-filename "bench_${result_prefix}.json" >> "$log_file" 2>&1
    local bench_status=$?

    kill_server $server_pid $gpu_id

    if [ $bench_status -eq 0 ]; then
        echo "[GPU $gpu_id] 完成: ${exp_type} ${scenario_name} ${param:+k*=$param}${run_info}"
    else
        echo "[GPU $gpu_id] 失败: ${exp_type} ${scenario_name}${run_info}"
    fi

    return $bench_status
}

# ========================================
# GPU Worker 并行调度
# ========================================
PROGRESS_FILE="${OUTPUT_DIR}/progress.txt"
LOCK_FILE="${OUTPUT_DIR}/.queue.lock"

gpu_worker() {
    local gpu_id=$1

    while true; do
        local exp=$(get_next_experiment "$QUEUE_FILE" "$LOCK_FILE")
        [ -z "$exp" ] && break

        # 解析实验参数 (格式: type|input_len|output_len|shape|hazard_type|param|run_num)
        IFS='|' read -r exp_type input_len output_len gamma_shape hazard_type param run_num <<< "$exp"

        if run_experiment "$gpu_id" "$exp_type" "$input_len" "$output_len" "$gamma_shape" "$hazard_type" "$param" "$run_num"; then
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
echo "分析结果:"
echo "  python reproduce/hazard_rate/analyze_hazard_rate.py ${OUTPUT_DIR}"
