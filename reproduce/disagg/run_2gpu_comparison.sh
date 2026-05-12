#!/bin/bash

# 2-GPU 公平对比: baseline(DP=2) vs pd_auto(DP=2) vs disagg(P/D分离)
#
# 用法: ./run_2gpu_comparison.sh [GPU1] [GPU2]
#
# 环境变量:
#   MODEL: 模型路径，默认 Qwen/Qwen3-8B
#   MAX_CONCURRENCY: 并发数，默认 64
#   NUM_PROMPTS: 请求数，默认 1000
#   INPUT_LEN / OUTPUT_LEN: 固定 input/output 长度
#   SKIP_DISAGG: 设为1跳过disagg (高并发下可能OOM)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common/common.sh"

GPU1=${1:-0}
GPU2=${2:-1}
MODEL=${MODEL:-"Qwen/Qwen3-8B"}
MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||')
MAX_CONCURRENCY=${MAX_CONCURRENCY:-64}
NUM_PROMPTS=${NUM_PROMPTS:-1000}
INPUT_LEN=${INPUT_LEN:-512}
OUTPUT_LEN=${OUTPUT_LEN:-256}
OUTPUT_VARIANCE=${OUTPUT_VARIANCE:-0.25}
SOURCE_DATASET=${SOURCE_DATASET:-"alpaca"}
K_RATIO=${K_RATIO:-0.8}
PORT=${PORT:-13000}
SKIP_DISAGG=${SKIP_DISAGG:-0}
KV_BUFFER_SIZE=${KV_BUFFER_SIZE:-2e10}

OUTPUT_DIR="${SCRIPT_DIR}/../outputs/2gpu_comparison_${MODEL_SHORT}_c${MAX_CONCURRENCY}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR/logs"

init_experiment_env

# 硬件校准
ensure_calibration "$MODEL" "$MODEL_SHORT"

echo "========================================"
echo "2-GPU 公平对比 (concurrency=${MAX_CONCURRENCY})"
echo "========================================"
echo "  MODEL: $MODEL"
echo "  GPUs: $GPU1, $GPU2"
echo "  NUM_PROMPTS: $NUM_PROMPTS"
echo "  INPUT_LEN: $INPUT_LEN, OUTPUT_LEN: $OUTPUT_LEN"
echo ""

# ========================================
# 生成数据集
# ========================================
DATASET="${OUTPUT_DIR}/synthetic.jsonl"
python3 "${SCRIPT_DIR}/generate_distribution_shift_dataset.py" \
    --model "$MODEL" \
    --num-prompts-per-phase "$NUM_PROMPTS" \
    --phases "${INPUT_LEN}:${OUTPUT_LEN}" \
    --variance "$OUTPUT_VARIANCE" \
    --source-dataset "$SOURCE_DATASET" \
    --output "$DATASET" \
    --seed 42

# 公共 bench 参数
bench_common=(
    --model "$MODEL"
    --dataset-name custom
    --dataset-path "$DATASET"
    --custom-output-len -1
    --ignore-eos
    --num-prompts "$NUM_PROMPTS"
    --num-warmups 0
    --request-rate inf
    --max-concurrency "$MAX_CONCURRENCY"
    --save-result
    --result-dir "$OUTPUT_DIR"
)

dtype_arg=""
if [ -n "${DTYPE:-}" ]; then
    dtype_arg="--dtype $DTYPE"
fi

# ========================================
# 辅助函数
# ========================================
run_dp2_bench() {
    local scheduler=$1
    local result_file=$2
    local log_file="${OUTPUT_DIR}/logs/${scheduler}.log"

    echo ""
    echo "--- ${scheduler} (DP=2) ---"

    # 清理
    lsof -t -i:$PORT 2>/dev/null | xargs -r kill -9 2>/dev/null
    wait_for_gpu_memory $GPU1 60 || return 1
    wait_for_gpu_memory $GPU2 60 || return 1

    # 设置环境变量
    local env_prefix="CUDA_VISIBLE_DEVICES=${GPU1},${GPU2}"
    case "$scheduler" in
        baseline)
            env_prefix="$env_prefix"
            ;;
        pd_auto)
            env_prefix="$env_prefix VLLM_PD_SCHEDULER_MODE=auto VLLM_PD_K_MODE=ratio VLLM_PD_K_RATIO=$K_RATIO VLLM_PD_CALIBRATION_FILE=$VLLM_PD_CALIBRATION_FILE"
            ;;
    esac

    env $env_prefix vllm serve "$MODEL" \
        --port $PORT --gpu-memory-utilization 0.9 --data-parallel-size 2 \
        $dtype_arg > "$log_file" 2>&1 &
    local pid=$!

    if ! wait_for_server $PORT $pid 300 "$log_file"; then
        echo "${scheduler} 启动失败"
        kill_server $pid
        return 1
    fi

    local status=0
    vllm bench serve "${bench_common[@]}" \
        --base-url "http://localhost:${PORT}" \
        --result-filename "$result_file" \
        >> "$log_file" 2>&1 || status=$?

    kill $pid 2>/dev/null; wait $pid 2>/dev/null

    [ $status -eq 0 ] && echo "${scheduler} 完成" || echo "${scheduler} 失败 (exit=$status)"
    return $status
}

run_disagg_bench() {
    local result_file=$1
    local log_dir="${OUTPUT_DIR}/logs"

    echo ""
    echo "--- disagg (P/D分离) ---"

    lsof -t -i:9000 -i:9100 -i:9200 -i:14579 -i:14580 2>/dev/null | xargs -r kill -9 2>/dev/null
    wait_for_gpu_memory $GPU1 60 || return 1
    wait_for_gpu_memory $GPU2 60 || return 1

    # 清理 PD 调度器环境变量，避免影响 disagg
    unset VLLM_PD_SCHEDULER_MODE VLLM_PD_K_MODE VLLM_PD_K_RATIO \
          VLLM_PD_CALIBRATION_FILE VLLM_USE_PD_SCHEDULER 2>/dev/null || true

    export VLLM_HOST_IP=127.0.0.1

    CUDA_VISIBLE_DEVICES=$GPU1 vllm serve "$MODEL" \
        --port 9100 --gpu-memory-utilization 0.8 $dtype_arg \
        --kv-transfer-config \
        '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2,"kv_buffer_size":'"$KV_BUFFER_SIZE"',"kv_port":14579}' \
        > "${log_dir}/disagg_prefill.log" 2>&1 &
    local prefill_pid=$!

    CUDA_VISIBLE_DEVICES=$GPU2 vllm serve "$MODEL" \
        --port 9200 --gpu-memory-utilization 0.8 $dtype_arg \
        --kv-transfer-config \
        '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2,"kv_buffer_size":'"$KV_BUFFER_SIZE"',"kv_port":14580}' \
        > "${log_dir}/disagg_decode.log" 2>&1 &
    local decode_pid=$!

    if ! wait_for_server 9100 $prefill_pid 300 "${log_dir}/disagg_prefill.log"; then
        echo "disagg prefill 启动失败"
        kill $prefill_pid $decode_pid 2>/dev/null
        return 1
    fi
    if ! wait_for_server 9200 $decode_pid 300 "${log_dir}/disagg_decode.log"; then
        echo "disagg decode 启动失败"
        kill $prefill_pid $decode_pid 2>/dev/null
        return 1
    fi

    python3 "${SCRIPT_DIR}/../../benchmarks/disagg_benchmarks/disagg_prefill_proxy_server.py" \
        --port 9000 \
        --prefill-url http://localhost:9100 \
        --decode-url http://localhost:9200 \
        > "${log_dir}/disagg_proxy.log" 2>&1 &
    local proxy_pid=$!
    sleep 3

    # 验证 proxy
    if ! curl -s --max-time 60 http://localhost:9000/v1/completions \
        -H "Content-Type: application/json" \
        -d '{"model":"'"$MODEL"'","prompt":"test","max_tokens":1}' >/dev/null 2>&1; then
        echo "disagg proxy 验证失败"
        kill $proxy_pid $prefill_pid $decode_pid 2>/dev/null
        return 1
    fi

    local status=0
    vllm bench serve "${bench_common[@]}" \
        --base-url "http://localhost:9000" \
        --result-filename "$result_file" \
        >> "${log_dir}/disagg_bench.log" 2>&1 || status=$?

    kill $proxy_pid 2>/dev/null
    kill_server $prefill_pid $GPU1
    kill_server $decode_pid $GPU2

    [ $status -eq 0 ] && echo "disagg 完成" || echo "disagg 失败 (exit=$status)"
    return $status
}

# ========================================
# 运行实验
# ========================================
run_dp2_bench "baseline" "bench_baseline.json" || echo "警告: baseline 失败"
run_dp2_bench "pd_auto" "bench_pd_auto.json" || echo "警告: pd_auto 失败"

if [ "$SKIP_DISAGG" != "1" ]; then
    run_disagg_bench "bench_disagg.json" || echo "警告: disagg 失败"
fi

# ========================================
# 汇总结果
# ========================================
echo ""
echo "========================================"
echo "结果汇总 (concurrency=${MAX_CONCURRENCY}, ${NUM_PROMPTS} prompts)"
echo "========================================"
echo ""
printf "%-15s %15s %15s %10s %10s\n" "Scheduler" "TotalThrput" "OutputThrput" "TTFT(ms)" "TPOT(ms)"
printf "%-15s %15s %15s %10s %10s\n" "----------" "-----------" "------------" "--------" "--------"

for f in "$OUTPUT_DIR"/bench_*.json; do
    [ -f "$f" ] || continue
    name=$(basename "$f" .json | sed 's/^bench_//')
    python3 -c "
import json, sys
name = sys.argv[1]
d = json.load(open(sys.argv[2]))
print(f'{name:<15s} {d[\"total_token_throughput\"]:15.2f} {d[\"output_throughput\"]:15.2f} {d[\"mean_ttft_ms\"]:10.2f} {d[\"mean_tpot_ms\"]:10.2f}')
" "$name" "$f" 2>/dev/null || true
done

echo ""
echo "结果目录: $OUTPUT_DIR"
