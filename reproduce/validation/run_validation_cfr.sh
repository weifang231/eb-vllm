#!/bin/bash
#
# CFR online-controller validation experiment.
#
# For each of the three CFR workloads, runs a single (B, N) configuration
# under EB(k̂) with the online estimator enabled.  The CFR update history
# (saved into *_stats.json) lets the analysis script compare:
#   (i)   p̂_0 vs ground-truth p_0 = 1/E[O];
#         μ̂_L vs ground-truth E[L];
#   (ii)  realised throughput vs the fluid-optimal TP(θ_0·N̂, N̂);
#   (iii) OOM rate (preempted requests / total) vs the prescribed ε.
#
# Usage:
#   ./run_validation_cfr.sh [MAX_GPUS]
#
# The (B, N) chosen here are sensible defaults for Qwen3-8B on H200 / RTX
# PRO 6000 — override with BS / TB env vars if you have better numbers
# from the grid search.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common/common_cfr.sh"

WORKER_PIDS=()
cleanup() {
    for pid in "${WORKER_PIDS[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM HUP

MAX_GPUS=${1:-3}
MODEL=${MODEL:-"Qwen/Qwen3-8B"}
NUM_PROMPTS=${NUM_PROMPTS:-4000}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-2048}
NUM_WARMUP_REQUESTS=${NUM_WARMUP_REQUESTS:-100}
RANDOM_RANGE_RATIO=${RANDOM_RANGE_RATIO:-0.5}
BASE_PORT=${BASE_PORT:-10100}
SKIP_EXISTING=${SKIP_EXISTING:-1}

# Per-workload (B, N) defaults — tuned for Qwen3-8B but override freely.
# Paper Figure 3 uses N=1024 uniformly across all three workloads (see caption
# "Validation on H200, N = 1024"); to match it exactly, set
#   BS_DECODE_HEAVY=BS_BALANCED=BS_PREFILL_HEAVY=1024.
TB_DECODE_HEAVY=${TB_DECODE_HEAVY:-16384};  BS_DECODE_HEAVY=${BS_DECODE_HEAVY:-2048}
TB_BALANCED=${TB_BALANCED:-14336};          BS_BALANCED=${BS_BALANCED:-1024}
TB_PREFILL_HEAVY=${TB_PREFILL_HEAVY:-18432}; BS_PREFILL_HEAVY=${BS_PREFILL_HEAVY:-512}

# Online-estimator pacing — match the paper's reported window sizes.
export VLLM_PD_PARAM_UPDATE_INTERVAL=${VLLM_PD_PARAM_UPDATE_INTERVAL:-100}
export VLLM_PD_IFR_UPDATE_INTERVAL=${VLLM_PD_IFR_UPDATE_INTERVAL:-100}
export VLLM_PD_IFR_WINDOW_SIZE=${VLLM_PD_IFR_WINDOW_SIZE:-500}
export VLLM_PD_IFR_MIN_SAMPLES=${VLLM_PD_IFR_MIN_SAMPLES:-50}
# Default to fixed N (no auto-shrink). Paper Figure 3 was generated at fixed
# N=1024; the memory-safe auto-compute (Proposition prop:memory) is conservative
# and yields significantly lower throughput than the figure suggests. Override
# with VLLM_PD_AUTO_COMPUTE_N=1 to enable the memory-safe online controller.
export VLLM_PD_AUTO_COMPUTE_N=${VLLM_PD_AUTO_COMPUTE_N:-0}
export VLLM_PD_OOM_TOLERANCE=${VLLM_PD_OOM_TOLERANCE:-0.01}

init_experiment_env
detect_gpu_name
resolve_calibration "$MODEL"
read_calibration_params

MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||')
OUTPUT_DIR=${OUTPUT_DIR:-"${SCRIPT_DIR}/outputs/controller_validation/${GPU_TAG}_${MODEL_SHORT}"}
mkdir -p "$OUTPUT_DIR"

select_gpus "$MAX_GPUS"

echo "========================================"
echo "CFR controller-validation experiment"
echo "========================================"
echo "  GPU: ${GPU_NAME} (${GPU_TAG}); MODEL: ${MODEL}"
echo "  ε (OOM tolerance): ${VLLM_PD_OOM_TOLERANCE}"
echo "  Update interval: ${VLLM_PD_IFR_UPDATE_INTERVAL}"
echo "  Window size: ${VLLM_PD_IFR_WINDOW_SIZE} (min samples=${VLLM_PD_IFR_MIN_SAMPLES})"
echo "  OUTPUT: ${OUTPUT_DIR}"
echo ""

# ----------------------------------------------------------------------
# Build queue: one (workload × scheduler) per row; we run v1 too as a
# reference TP for the "throughput attainment" column.
# ----------------------------------------------------------------------
QUEUE_FILE="${OUTPUT_DIR}/experiment_queue.txt"
> "$QUEUE_FILE"
for scenario in decode_heavy balanced prefill_heavy; do
    case "$scenario" in
        decode_heavy)   tb=$TB_DECODE_HEAVY;   bs=$BS_DECODE_HEAVY ;;
        balanced)       tb=$TB_BALANCED;       bs=$BS_BALANCED ;;
        prefill_heavy)  tb=$TB_PREFILL_HEAVY;  bs=$BS_PREFILL_HEAVY ;;
    esac
    echo "v1|${scenario}|${bs}|${tb}"     >> "$QUEUE_FILE"
    echo "eb_khat|${scenario}|${bs}|${tb}" >> "$QUEUE_FILE"
done
TOTAL_EXPERIMENTS=$(wc -l < "$QUEUE_FILE")

cat > "${OUTPUT_DIR}/experiment_config.json" <<EOF
{
    "experiment_type": "cfr_controller_validation",
    "gpu_name": "${GPU_NAME}",
    "gpu_tag": "${GPU_TAG}",
    "model": "${MODEL}",
    "num_prompts": ${NUM_PROMPTS},
    "max_concurrency": ${MAX_CONCURRENCY},
    "num_warmup_requests": ${NUM_WARMUP_REQUESTS},
    "ifr_update_interval": ${VLLM_PD_IFR_UPDATE_INTERVAL},
    "ifr_window_size": ${VLLM_PD_IFR_WINDOW_SIZE},
    "ifr_min_samples": ${VLLM_PD_IFR_MIN_SAMPLES},
    "auto_compute_n": ${VLLM_PD_AUTO_COMPUTE_N},
    "oom_tolerance": ${VLLM_PD_OOM_TOLERANCE},
    "scenarios": {
        "decode_heavy":   {"input_len": 128,  "output_len": 1024, "bs": ${BS_DECODE_HEAVY},  "tb": ${TB_DECODE_HEAVY}},
        "balanced":       {"input_len": 512,  "output_len": 512,  "bs": ${BS_BALANCED},      "tb": ${TB_BALANCED}},
        "prefill_heavy":  {"input_len": 1024, "output_len": 128,  "bs": ${BS_PREFILL_HEAVY}, "tb": ${TB_PREFILL_HEAVY}}
    },
    "calibration_file": "${VLLM_PD_CALIBRATION_FILE}",
    "calibration_params": {
        "alpha_p": ${ALPHA_P}, "beta_p": ${BETA_P},
        "alpha_d": ${ALPHA_D}, "beta_d": ${BETA_D}
    },
    "total_experiments": ${TOTAL_EXPERIMENTS},
    "timestamp": "$(date -Iseconds)"
}
EOF

run_experiment() {
    local gpu_id=$1 sched=$2 scenario=$3 bs=$4 tb=$5
    set_cfr_scenario "$scenario"
    local port=$((BASE_PORT + gpu_id))
    local result_dir="${OUTPUT_DIR}/${scenario}_in${INPUT_LEN}_out${OUTPUT_LEN}"
    local result_file="${result_dir}/bench_${sched}.json"
    local log_file="${result_dir}/logs/${sched}.log"

    if [ "$SKIP_EXISTING" = "1" ] && [ -f "$result_file" ]; then
        echo "[GPU $gpu_id] SKIP ${sched} ${scenario}"
        return 0
    fi
    mkdir -p "${result_dir}/logs"
    : > "$log_file"
    check_port_available "$port" "$gpu_id" || return 1

    echo "[GPU $gpu_id] START ${sched} ${scenario} bs=${bs} tb=${tb}"

    export CUDA_VISIBLE_DEVICES=$gpu_id
    export VLLM_COLLECT_SCHEDULE_STATS=1
    set_scheduler_env "$sched" || return 1

    wait_for_gpu_memory "$gpu_id" 60 || return 1

    VLLM_SCHEDULE_STATS_FILE="${result_dir}/${sched}_stats.json" \
    vllm serve "$MODEL" \
        --port "$port" \
        --gpu-memory-utilization 0.9 \
        --max-num-seqs "$bs" \
        --max-num-batched-tokens "$tb" >> "$log_file" 2>&1 &
    local server_pid=$!

    if ! wait_for_server "$port" "$server_pid" 240 "$log_file"; then
        echo "[GPU $gpu_id] FAIL: server didn't start"
        kill_server "$server_pid" "$gpu_id"
        return 1
    fi

    vllm bench serve \
        --model "$MODEL" \
        --base-url "http://localhost:${port}" \
        --dataset-name geometric_random \
        --random-input-len "$INPUT_LEN" \
        --random-output-len "$OUTPUT_LEN" \
        --random-range-ratio "$RANDOM_RANGE_RATIO" \
        --num-prompts "$NUM_PROMPTS" \
        --num-warmups "$NUM_WARMUP_REQUESTS" \
        --request-rate inf \
        --max-concurrency "$MAX_CONCURRENCY" \
        --save-result \
        --result-dir "${result_dir}" \
        --result-filename "bench_${sched}.json" >> "$log_file" 2>&1
    local bench_status=$?

    kill_server "$server_pid" "$gpu_id"

    if [ $bench_status -eq 0 ]; then
        echo "[GPU $gpu_id] DONE  ${sched} ${scenario}"
    else
        echo "[GPU $gpu_id] FAIL  ${sched} ${scenario}"
    fi
    return $bench_status
}

PROGRESS_FILE="${OUTPUT_DIR}/progress.txt"
LOCK_FILE="${OUTPUT_DIR}/.queue.lock"
> "$PROGRESS_FILE"

gpu_worker() {
    local gpu_id=$1
    while true; do
        local exp
        exp=$(get_next_experiment "$QUEUE_FILE" "$LOCK_FILE")
        [ -z "$exp" ] && break
        IFS='|' read -r sched scenario bs tb <<< "$exp"
        if run_experiment "$gpu_id" "$sched" "$scenario" "$bs" "$tb"; then
            update_progress "OK|${exp}" "$PROGRESS_FILE" "$LOCK_FILE" "$TOTAL_EXPERIMENTS"
        else
            update_progress "FAIL|${exp}" "$PROGRESS_FILE" "$LOCK_FILE" "$TOTAL_EXPERIMENTS"
        fi
    done
}

for gpu_id in "${GPUS_TO_USE[@]}"; do
    gpu_worker "$gpu_id" &
    WORKER_PIDS+=($!)
    sleep 10
done
for pid in "${WORKER_PIDS[@]}"; do wait "$pid" || true; done

print_summary "$PROGRESS_FILE" "$TOTAL_EXPERIMENTS" "$OUTPUT_DIR"
echo ""
echo "Analyse with:"
echo "  python ${SCRIPT_DIR}/analyze_cfr_validation.py ${OUTPUT_DIR}"
