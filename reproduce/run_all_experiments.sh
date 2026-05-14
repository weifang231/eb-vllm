#!/bin/bash

# One-command script to run all experiments
# Usage: ./run_all_experiments.sh <MODEL> [MAX_GPUS]
#
# Examples:
#   ./reproduce/run_all_experiments.sh Qwen/Qwen3-8B 4
#   ./reproduce/run_all_experiments.sh meta-llama/Llama-3.1-8B 2

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check arguments
if [ -z "${1:-}" ]; then
    echo "Usage: $0 <MODEL> [MAX_GPUS]"
    echo ""
    echo "Arguments:"
    echo "  MODEL      model name (e.g. Qwen/Qwen3-8B)"
    echo "  MAX_GPUS   number of GPUs to use (default 4)"
    echo ""
    echo "Examples:"
    echo "  $0 Qwen/Qwen3-8B 4"
    echo "  $0 meta-llama/Llama-3.1-8B 2"
    echo ""
    echo "Environment variables (optional):"
    echo "  SKIP_CALIBRATION=true   skip calibration step"
    echo "  SKIP_EXPORT=true        skip dataset export step"
    echo "  EXPERIMENTS=\"sharegpt numina_math\"  only run specified experiments"
    echo "  SCHEDULERS=\"baseline pd_ratio pd_ifr\"  only run specified scheduler modes"
    echo "  SKIP_EXISTING=0         do not skip existing results (default: skip)"
    echo "  VERSION=1               filename suffix, used for repeated runs to produce ifr_1, ifr_2, etc."
    echo ""
    echo "Scheduler modes (SCHEDULERS):"
    echo "  baseline   vLLM default scheduler"
    echo "  pd_ratio   PD scheduler with ratio mode (θ*=K_RATIO)"
    echo "  pd_ifr     PD scheduler with IFR mode (adaptive θ* based on hazard rate)"
    echo ""
    echo "Example - run only pd_ifr mode experiments:"
    echo "  SCHEDULERS=pd_ifr $0 Qwen/Qwen3-8B 4"
    exit 1
fi

MODEL="$1"
MAX_GPUS=${2:-4}
MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||')

echo "========================================"
echo "PD Scheduler full experiment suite"
echo "========================================"
echo ""
echo "Configuration:"
echo "  MODEL: $MODEL"
echo "  MODEL_SHORT: $MODEL_SHORT"
echo "  MAX_GPUS: $MAX_GPUS"
echo ""

# Create output directory
mkdir -p "${SCRIPT_DIR}/outputs"

# Record start time
START_TIME=$(date +%s)
log_time() {
    local now=$(date +%s)
    local elapsed=$((now - START_TIME))
    local hours=$((elapsed / 3600))
    local minutes=$(((elapsed % 3600) / 60))
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [${hours}h ${minutes}m] $1"
}

# ============================================================
# Step 0: Hardware calibration
# ============================================================
CALIBRATION_FILE="${SCRIPT_DIR}/outputs/pd_calibration_${MODEL_SHORT}.json"

if [ "${SKIP_CALIBRATION:-false}" = "true" ]; then
    log_time "Skipping calibration step (SKIP_CALIBRATION=true)"
elif [ -f "$CALIBRATION_FILE" ]; then
    log_time "Calibration file already exists: $CALIBRATION_FILE"
    echo "  To re-calibrate, delete this file or set SKIP_CALIBRATION=false"
else
    log_time "Starting hardware calibration..."
    # Some models do not support float16; specify via DTYPE=bfloat16
    DTYPE_ARG=""
    if [ -n "${DTYPE:-}" ]; then
        DTYPE_ARG="--dtype $DTYPE"
    fi
    python -m vllm.v1.core.sched.calibration --model "$MODEL" $DTYPE_ARG --output "$CALIBRATION_FILE"
    log_time "Calibration done: $CALIBRATION_FILE"
fi

# Validate calibration file
if [ ! -f "$CALIBRATION_FILE" ]; then
    echo "Error: calibration file not found: $CALIBRATION_FILE"
    exit 1
fi

export VLLM_PD_CALIBRATION_FILE="$CALIBRATION_FILE"

# SKIP_EXISTING: skip existing result files by default (convenient when running only one scheduler mode)
export SKIP_EXISTING=${SKIP_EXISTING:-1}
echo "  SKIP_EXISTING: $SKIP_EXISTING"
echo ""

# ============================================================
# Step 1: Export datasets
# ============================================================
log_time "Preparing datasets..."

SHAREGPT_PROMPTS="${SCRIPT_DIR}/outputs/sharegpt_prompts.jsonl"
NUMINA_PROMPTS="${SCRIPT_DIR}/outputs/numina_math_prompts.jsonl"
LONGBENCH_PROMPTS="${SCRIPT_DIR}/outputs/longbench_prefill.jsonl"
WILDCHAT_DATA="${SCRIPT_DIR}/outputs/wildchat_multiturn.json"

if [ "${SKIP_EXPORT:-false}" = "true" ]; then
    log_time "Skipping dataset export (SKIP_EXPORT=true)"
else
    # ShareGPT
    if [ ! -f "$SHAREGPT_PROMPTS" ]; then
        log_time "Exporting ShareGPT dataset..."
        # Download raw data
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
        log_time "ShareGPT dataset already exists"
    fi

    # numina_math
    if [ ! -f "$NUMINA_PROMPTS" ]; then
        log_time "Exporting numina_math dataset..."
        python "${SCRIPT_DIR}/common/export_dataset.py" \
            --dataset numina_math \
            --model "$MODEL" \
            --num-samples 4000 \
            --min-output-len 800 \
            --output "$NUMINA_PROMPTS"
    else
        log_time "numina_math dataset already exists"
    fi

    # longbench
    if [ ! -f "$LONGBENCH_PROMPTS" ]; then
        log_time "Exporting longbench dataset..."
        python "${SCRIPT_DIR}/common/export_dataset.py" \
            --dataset longbench \
            --model "$MODEL" \
            --num-samples 4000 \
            --min-input-len 1000 \
            --max-input-len 4000 \
            --output "$LONGBENCH_PROMPTS"
    else
        log_time "longbench dataset already exists"
    fi

    # WildChat (multi-turn conversations)
    if [ ! -f "$WILDCHAT_DATA" ]; then
        log_time "Exporting WildChat multi-turn dataset..."
        python "${SCRIPT_DIR}/real_workloads/multiturn/export_dataset.py" \
            --dataset wildchat \
            --model "$MODEL" \
            --num-conversations 3000 \
            --min-turns 6 \
            --output "$WILDCHAT_DATA"
    else
        log_time "WildChat dataset already exists"
    fi
fi

echo ""

# ============================================================
# Step 2: Run experiments
# ============================================================

# Run all experiments by default
EXPERIMENTS=${EXPERIMENTS:-"sharegpt numina_math longbench wildchat"}

run_experiment() {
    local name=$1
    local dataset=$2
    local output_len=$3
    local enable_thinking=$4
    local script=$5

    log_time "Starting experiment: $name"
    echo "  Dataset: $dataset"
    echo "  Output length: $output_len"
    echo "  Thinking: $enable_thinking"
    echo ""

    ENABLE_THINKING=$enable_thinking \
    CUSTOM_OUTPUT_LEN=$output_len \
    MODEL=$MODEL \
    DTYPE=${DTYPE:-} \
        "$script" "$dataset" "$MAX_GPUS"

    log_time "Experiment done: $name"
    echo ""
}

# ShareGPT: balanced workload
if [[ "$EXPERIMENTS" == *"sharegpt"* ]]; then
    if [ -f "$SHAREGPT_PROMPTS" ]; then
        run_experiment "ShareGPT" "$SHAREGPT_PROMPTS" 500 false "${SCRIPT_DIR}/real_workloads/run_grid_search.sh"
    else
        log_time "Skipping ShareGPT (dataset not found)"
    fi
fi

# numina_math: decode-heavy
if [[ "$EXPERIMENTS" == *"numina_math"* ]]; then
    if [ -f "$NUMINA_PROMPTS" ]; then
        run_experiment "numina_math" "$NUMINA_PROMPTS" 4000 true "${SCRIPT_DIR}/real_workloads/run_grid_search.sh"
    else
        log_time "Skipping numina_math (dataset not found)"
    fi
fi

# longbench: prefill-heavy
if [[ "$EXPERIMENTS" == *"longbench"* ]]; then
    if [ -f "$LONGBENCH_PROMPTS" ]; then
        run_experiment "longbench" "$LONGBENCH_PROMPTS" 20 false "${SCRIPT_DIR}/real_workloads/run_grid_search.sh"
    else
        log_time "Skipping longbench (dataset not found)"
    fi
fi

# WildChat: multi-turn conversation (prefix cache)
if [[ "$EXPERIMENTS" == *"wildchat"* ]]; then
    if [ -f "$WILDCHAT_DATA" ]; then
        log_time "Starting experiment: WildChat (multi-turn)"
        echo "  Dataset: $WILDCHAT_DATA"
        echo ""

        MODEL=$MODEL \
        DTYPE=${DTYPE:-} \
            "${SCRIPT_DIR}/real_workloads/multiturn/run_benchmark.sh" "$WILDCHAT_DATA" "$MAX_GPUS"

        log_time "Experiment done: WildChat"
        echo ""
    else
        log_time "Skipping WildChat (dataset not found)"
    fi
fi

# ============================================================
# Step 3: Summarize results
# ============================================================
echo ""
echo "========================================"
log_time "All experiments completed!"
echo "========================================"
echo ""
echo "Result directories:"

# List generated result directories
for dir in "${SCRIPT_DIR}/outputs/grid_search_"*"_${MODEL_SHORT}_"* "${SCRIPT_DIR}/outputs/multiturn_"*"_${MODEL_SHORT}_"*; do
    if [ -d "$dir" ]; then
        echo "  $dir"
    fi
done

echo ""
echo "Run analysis scripts:"
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
    echo "# WildChat (multi-turn)"
    echo "python ${SCRIPT_DIR}/multiturn/analyze_results.py $WILDCHAT_DIR"
    echo ""
fi

# Compute total elapsed time
END_TIME=$(date +%s)
TOTAL_ELAPSED=$((END_TIME - START_TIME))
TOTAL_HOURS=$((TOTAL_ELAPSED / 3600))
TOTAL_MINUTES=$(((TOTAL_ELAPSED % 3600) / 60))
echo "========================================"
echo "Total elapsed: ${TOTAL_HOURS}h ${TOTAL_MINUTES}m"
echo "========================================"
