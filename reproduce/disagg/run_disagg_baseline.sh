#!/bin/bash

# vLLM Disaggregated Prefill 基准测试
# 使用 vLLM 官方的 P/D disaggregation 方案 (2 GPU: 1 prefill + 1 decode)
# 与我们的单 GPU PD scheduler 做对比
#
# 架构:
#   GPU 0: prefill instance (kv_producer, port 8100)
#   GPU 1: decode instance  (kv_consumer, port 8200)
#   Proxy: port 8000, 路由请求到 prefill → decode
#
# 用法: ./run_disagg_baseline.sh [PREFILL_GPU] [DECODE_GPU]
#
# 环境变量:
#   MODEL: 模型路径，默认 Qwen/Qwen3-8B
#   CONCURRENCY_PHASES: 并发阶段，格式同 run_concurrency_shift.sh
#   INPUT_LEN / OUTPUT_LEN: 固定 input/output 长度
#   PROXY_PORT: proxy 端口，默认 8000
#   PREFILL_PORT / DECODE_PORT: prefill/decode 实例端口

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common/common.sh"

# GPU 分配
PREFILL_GPU=${1:-0}
DECODE_GPU=${2:-1}

# 实验参数
MODEL=${MODEL:-"Qwen/Qwen3-8B"}
MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||')
NUM_PROMPTS_PER_PHASE=${NUM_PROMPTS_PER_PHASE:-2000}
CONCURRENCY_PHASES=${CONCURRENCY_PHASES:-"32:1000,2048:3000,256:2000"}
INPUT_LEN=${INPUT_LEN:-512}
OUTPUT_LEN=${OUTPUT_LEN:-256}
OUTPUT_VARIANCE=${OUTPUT_VARIANCE:-0.25}
SOURCE_DATASET=${SOURCE_DATASET:-"alpaca"}

# 端口配置
PROXY_PORT=${PROXY_PORT:-9000}
PREFILL_PORT=${PREFILL_PORT:-9100}
DECODE_PORT=${DECODE_PORT:-9200}

# 服务配置
TB=${TB:-18432}
BS=${BS:-2048}
KV_BUFFER_SIZE=${KV_BUFFER_SIZE:-2e10}

# 解析并发阶段
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

# 输出目录
OUTPUT_DIR="${SCRIPT_DIR}/../outputs/disagg_baseline_${MODEL_SHORT}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR/logs"

# 初始化环境
init_experiment_env

export VLLM_HOST_IP=${VLLM_HOST_IP:-127.0.0.1}

echo "========================================"
echo "Disaggregated Prefill 基准测试 (${NUM_PHASES}-phase)"
echo "========================================"
echo ""
echo "实验配置:"
echo "  MODEL: $MODEL"
echo "  Prefill GPU: $PREFILL_GPU, Decode GPU: $DECODE_GPU"
echo "  TB: $TB, BS: $BS"
echo "  CONCURRENCY_PHASES: $CONCURRENCY_PHASES"
echo "  INPUT_LEN: $INPUT_LEN, OUTPUT_LEN: $OUTPUT_LEN"
echo "  Ports: proxy=$PROXY_PORT, prefill=$PREFILL_PORT, decode=$DECODE_PORT"
echo ""

# ========================================
# Step 1: 生成合成数据集 (与 concurrency_shift 相同)
# ========================================
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

# 构建 phases JSON
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
    "experiment_type": "disagg_baseline",
    "purpose": "vLLM disaggregated prefill baseline (2 GPU) for comparison with single-GPU PD scheduler",
    "model": "${MODEL}",
    "prefill_gpu": ${PREFILL_GPU},
    "decode_gpu": ${DECODE_GPU},
    "tb": ${TB},
    "bs": ${BS},
    "num_phases": ${NUM_PHASES},
    "concurrency_phases": ${PHASES_JSON},
    "input_len": ${INPUT_LEN},
    "output_len": ${OUTPUT_LEN},
    "output_variance": ${OUTPUT_VARIANCE},
    "kv_buffer_size": "${KV_BUFFER_SIZE}",
    "timestamp": "$(date -Iseconds)"
}
EOF

# ========================================
# Step 2: 启动 disagg 服务
# ========================================
PREFILL_LOG="${OUTPUT_DIR}/logs/prefill.log"
DECODE_LOG="${OUTPUT_DIR}/logs/decode.log"
PROXY_LOG="${OUTPUT_DIR}/logs/proxy.log"

cleanup_disagg() {
    echo "清理 disagg 服务..."
    [ -n "${PROXY_PID:-}" ] && kill $PROXY_PID 2>/dev/null || true
    [ -n "${PREFILL_PID:-}" ] && kill_server $PREFILL_PID $PREFILL_GPU
    [ -n "${DECODE_PID:-}" ] && kill_server $DECODE_PID $DECODE_GPU
}
trap cleanup_disagg EXIT

echo "启动 prefill instance (GPU $PREFILL_GPU, port $PREFILL_PORT)..."

local_dtype_arg=""
if [ -n "${DTYPE:-}" ]; then
    local_dtype_arg="--dtype $DTYPE"
fi

CUDA_VISIBLE_DEVICES=$PREFILL_GPU vllm serve "$MODEL" \
    --port $PREFILL_PORT \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs "$BS" \
    --max-num-batched-tokens "$TB" \
    $local_dtype_arg \
    --kv-transfer-config \
    '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2,"kv_buffer_size":'"$KV_BUFFER_SIZE"',"kv_port":14579}' \
    >> "$PREFILL_LOG" 2>&1 &
PREFILL_PID=$!

echo "启动 decode instance (GPU $DECODE_GPU, port $DECODE_PORT)..."

CUDA_VISIBLE_DEVICES=$DECODE_GPU vllm serve "$MODEL" \
    --port $DECODE_PORT \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs "$BS" \
    --max-num-batched-tokens "$TB" \
    $local_dtype_arg \
    --kv-transfer-config \
    '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2,"kv_buffer_size":'"$KV_BUFFER_SIZE"',"kv_port":14580}' \
    >> "$DECODE_LOG" 2>&1 &
DECODE_PID=$!

echo "等待 prefill instance 启动..."
if ! wait_for_server $PREFILL_PORT $PREFILL_PID 300 "$PREFILL_LOG"; then
    echo "Prefill instance 启动失败"
    exit 1
fi

echo "等待 decode instance 启动..."
if ! wait_for_server $DECODE_PORT $DECODE_PID 300 "$DECODE_LOG"; then
    echo "Decode instance 启动失败"
    exit 1
fi

echo "启动 proxy server (port $PROXY_PORT)..."

# 确保 proxy 端口可用
if lsof -nP -iTCP:$PROXY_PORT -sTCP:LISTEN >/dev/null 2>&1; then
    echo "警告: 端口 $PROXY_PORT 被占用，尝试清理..."
    lsof -t -i:$PROXY_PORT | xargs -r kill -9 2>/dev/null
    sleep 2
fi

PROXY_SCRIPT="${SCRIPT_DIR}/../../benchmarks/disagg_benchmarks/disagg_prefill_proxy_server.py"

python3 "$PROXY_SCRIPT" \
    --port $PROXY_PORT \
    --prefill-url "http://localhost:${PREFILL_PORT}" \
    --decode-url "http://localhost:${DECODE_PORT}" \
    >> "$PROXY_LOG" 2>&1 &
PROXY_PID=$!

# 等待 proxy 启动
echo "等待 proxy server 启动..."
local_i=0
while [ $local_i -lt 30 ]; do
    if curl -s "http://localhost:${PROXY_PORT}/" >/dev/null 2>&1; then
        break
    fi
    if ! kill -0 $PROXY_PID 2>/dev/null; then
        echo "Proxy server 进程退出:"
        cat "$PROXY_LOG"
        exit 1
    fi
    sleep 1
    local_i=$((local_i + 1))
done

if [ $local_i -ge 30 ]; then
    echo "Proxy server 启动超时"
    cat "$PROXY_LOG"
    exit 1
fi

echo "所有服务已启动"
echo "  Prefill: PID=$PREFILL_PID, port=$PREFILL_PORT"
echo "  Decode:  PID=$DECODE_PID, port=$DECODE_PORT"
echo "  Proxy:   PID=$PROXY_PID, port=$PROXY_PORT"

# ========================================
# Step 3: 运行 benchmark (与 concurrency_shift 相同的 phases)
# ========================================
overall_status=0

for phase_idx_0 in $(seq 0 $((NUM_PHASES - 1))); do
    concurrency=${PHASE_CONCURRENCIES[$phase_idx_0]}
    phase_prompts=${PHASE_NUM_PROMPTS[$phase_idx_0]}
    phase_idx=$((phase_idx_0 + 1))

    echo ""
    echo "--- Phase ${phase_idx}/${NUM_PHASES}: concurrency=${concurrency}, num_prompts=${phase_prompts} ---"

    bench_status=0
    vllm bench serve \
        --model "$MODEL" \
        --base-url "http://localhost:${PROXY_PORT}" \
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
        --result-filename "bench_disagg_phase${phase_idx}_c${concurrency}.json" \
        >> "${OUTPUT_DIR}/logs/benchmark.log" 2>&1 || bench_status=$?

    if [ $bench_status -eq 0 ]; then
        echo "Phase ${phase_idx} 完成 (concurrency=${concurrency}, prompts=${phase_prompts})"
    else
        echo "Phase ${phase_idx} 失败 (concurrency=${concurrency}, exit=$bench_status)"
        overall_status=$bench_status
    fi
done

echo ""
echo "========================================"
if [ $overall_status -eq 0 ]; then
    echo "实验完成!"
else
    echo "实验部分失败!"
fi
echo "========================================"
echo ""
echo "结果目录: $OUTPUT_DIR"
