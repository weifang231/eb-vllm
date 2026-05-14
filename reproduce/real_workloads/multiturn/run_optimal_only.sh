#!/bin/bash
#
# Multi-turn variant of reproduce/real_workloads/run_optimal_only.sh: invokes
# multiturn/run_benchmark.sh once per scheduler with its paper-reported optimal
# (B, N) from Appendix Table tab:optimal-config-h200 (H200, Qwen3-8B).
#
# Scheduler -> paper-vocab mapping (see evaluation.tex §4.3.1):
#   baseline = v1 (vLLM default mixed batching)
#   pd_ratio = v0 (exclusive batching, EB(k=1) under heavy traffic)
#   pd_ifr   = EB(k_hat^*) (adaptive threshold)
#
# All three schedulers may have *different* optimal (B, N), so each runs in its
# own run_benchmark.sh invocation (single-cell queue per call).  Output dir is
# shared, so analyze_results.py sees all three.
#
# Usage:
#   ./run_optimal_only.sh <DATASET_PATH> [NUM_GPUS]

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "${1:-}" ]; then
    echo "Usage: $0 <DATASET_PATH> [NUM_GPUS]"
    exit 1
fi
DATASET_PATH="$1"
NUM_GPUS=${2:-3}

DATASET_NAME=$(basename "$DATASET_PATH" .json)
case "$DATASET_NAME" in
    *wildchat*) WORKLOAD=wildchat ;;
    *)          echo "Error: only WildChat is supported for the multi-turn"
                echo "optimal-only path so far (got '$DATASET_NAME')."
                exit 1 ;;
esac

# Auto-detect GPU tag for calibration + (B, N) lookup.
detect_gpu_tag() {
    local name
    name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    case "$name" in
        *H200*)             echo "H200" ;;
        *"RTX PRO 6000"*|*RTXPRO6000*|*RTX6000*) echo "RTXPRO6000" ;;
        *L40S*)             echo "L40S" ;;
        *B300*)             echo "B300" ;;
        *)                  echo "${GPU_TAG:-H200}" ;;
    esac
}
GPU_TAG_DETECTED=$(detect_gpu_tag)

# Model dimension (paper has Qwen3-8B and Qwen3-30B-A3B on H200).
MODEL=${MODEL:-"Qwen/Qwen3-8B"}
MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||')
MODEL_TAG=${MODEL_TAG:-$MODEL_SHORT}

# Resolve per-(GPU, model) calibration if not already set.
if [ -z "${VLLM_PD_CALIBRATION_FILE:-}" ]; then
    CALIB="${SCRIPT_DIR}/../../calibration/pd_calibration_${MODEL_TAG}_${GPU_TAG_DETECTED}.json"
    if [ -f "$CALIB" ]; then
        export VLLM_PD_CALIBRATION_FILE="$CALIB"
        echo "Calibration: $VLLM_PD_CALIBRATION_FILE"
    else
        echo "Error: calibration file not found: $CALIB"
        echo "Run:  python -m vllm.v1.core.sched.calibration --model $MODEL --output $CALIB"
        exit 1
    fi
fi

# WildChat optimal (B, N) per (GPU, model, scheduler), from paper Appendix tables.
# Format: "B N"
case "$GPU_TAG_DETECTED:$MODEL_TAG:$WORKLOAD" in
    H200:Qwen3-8B:wildchat)
        BN_BASELINE="4096 2048"     # v1     (tab:optimal-config-h200)
        BN_PD_RATIO="18432 1536"    # v0
        BN_PD_IFR="16384 1024"      # EB(k_hat^*)
        ;;
    H200:Qwen3-30B-A3B:wildchat)
        BN_BASELINE="4096 1536"     # v1     (tab:optimal-config-h200)
        BN_PD_RATIO="16384 1024"    # v0
        BN_PD_IFR="14336 1024"      # EB(k_hat^*)
        ;;
    RTXPRO6000:Qwen3-8B:wildchat)
        BN_BASELINE="18432 1024"    # v1     (tab:optimal-config-a6000)
        BN_PD_RATIO="18432 1024"    # v0
        BN_PD_IFR="10240 1024"      # EB(k_hat^*)
        ;;
    *) echo "Error: unsupported (GPU=$GPU_TAG_DETECTED, model=$MODEL_TAG, workload=$WORKLOAD)"; exit 1 ;;
esac

SCHEDULERS=${SCHEDULERS:-"baseline pd_ifr"}

echo "Optimal-only multi-turn run"
echo "  workload   : $WORKLOAD"
echo "  dataset    : $DATASET_PATH"
echo "  gpus       : $NUM_GPUS"
echo "  schedulers : $SCHEDULERS"
echo "  baseline (v1)         B,N = ${BN_BASELINE}"
echo "  pd_ratio (v0)         B,N = ${BN_PD_RATIO}"
echo "  pd_ifr   (EB(k_hat*)) B,N = ${BN_PD_IFR}"
echo

run_one() {
    local label=$1 sched=$2 bn=$3
    read -r tb bs <<< "$bn"
    echo "==== ${label}: SCHEDULER='${sched}' TB=${tb} BS=${bs} ===="
    SCHEDULERS="$sched" BS_VALUES="$bs" TB_VALUES="$tb" \
        bash "${SCRIPT_DIR}/run_benchmark.sh" "$DATASET_PATH" "$NUM_GPUS"
}

# Iterate user-selected schedulers (default: baseline + pd_ifr; add pd_ratio
# via  SCHEDULERS="baseline pd_ratio pd_ifr" ./run_optimal_only.sh ...).
for sched in $SCHEDULERS; do
    case "$sched" in
        baseline) run_one "baseline (v1)"        baseline "$BN_BASELINE" ;;
        pd_ratio) run_one "pd_ratio (v0)"        pd_ratio "$BN_PD_RATIO" ;;
        pd_ifr)   run_one "pd_ifr   (EB(k_hat*))" pd_ifr  "$BN_PD_IFR" ;;
        *) echo "Error: unknown scheduler '$sched' (expected baseline/pd_ratio/pd_ifr)"; exit 1 ;;
    esac
done
