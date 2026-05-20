# Reproducing the EB(k̂*) paper (ICML 2026)

This directory contains the experiment scripts, analysis tools, and plot
scripts that reproduce every figure and table in the paper. Each
`secX_Y_*/` subdirectory corresponds to a specific paper section.

## Quick Start

```bash
# 1. Clone + setup environment
git clone https://github.com/weifang231/eb-vllm
cd eb-vllm
# `main` is the default branch and tracks the camera-ready snapshot.
# For the exact reviewed code, also: `git checkout v-icml2026-cr-rc1`
python -m venv .venv && source .venv/bin/activate     # OR: conda create -n eb python=3.12
pip install torch                                       # match your CUDA
VLLM_USE_PRECOMPILED=1 pip install -e . --no-build-isolation
                                # ~30 s; fetches all 8 .so (incl. FA3) from
                                # the official vLLM wheel — see warning below
pip install -r requirements/reproduce.txt              # datasets, aiohttp, quart, matplotlib, pandas, scipy

# 2. Download ShareGPT (other datasets auto-download from HF on first use)
wget -P reproduce/ \
    https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

# 3. Export prompt datasets for paper §4.3 real-world workloads (~5 min CPU)
cd reproduce/common
python export_dataset.py --dataset sharegpt    --model Qwen/Qwen3-8B --num-samples 4000 \
    --sharegpt-path ../ShareGPT_V3_unfiltered_cleaned_split.json \
    --output ../outputs/sharegpt_prompts.jsonl
python export_dataset.py --dataset longbench   --model Qwen/Qwen3-8B --num-samples 4000 \
    --min-input-len 1000 --max-input-len 4000 \
    --output ../outputs/longbench_prefill.jsonl
python export_dataset.py --dataset numina_math --model Qwen/Qwen3-8B --num-samples 4000 \
    --output ../outputs/numina_math_prompts.jsonl
cd ../real_workloads/multiturn
python export_dataset.py --dataset wildchat    --model Qwen/Qwen3-8B \
    --output ../../outputs/wildchat_multiturn.json     # defaults to paper-faithful 3000 conv / min_turns=6

# 4. One-time per-(model, GPU) cost-model calibration (~15 min per model)
cd ../../..   # back to eb-vllm/
python -m vllm.v1.core.sched.calibration --model Qwen/Qwen3-8B \
    --output reproduce/calibration/pd_calibration_Qwen3-8B_$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | sed 's/.*RTX PRO 6000.*/RTXPRO6000/;s/.*H200.*/H200/;s/ /_/g').json
# Sample H200 + RTX PRO 6000 calibrations are shipped under reproduce/calibration/.

# 5. Run one paper artifact (example: Table 2 Qwen3-8B ShareGPT)
cd reproduce/real_workloads
VLLM_PD_CALIBRATION_FILE=$PWD/../calibration/pd_calibration_Qwen3-8B_RTXPRO6000.json \
GPUS=0,1 SCHEDULERS="v1 eb" MODEL=Qwen/Qwen3-8B \
    bash run_optimal_only.sh ../outputs/sharegpt_prompts.jsonl 2

# 6. Analyze / plot
python analyze_grid_search.py ../outputs/optimal_only_sharegpt_prompts_Qwen3-8B_*
```

> **⚠ Do NOT drop `VLLM_USE_PRECOMPILED=1`.** A locally-compiled
> `vllm/vllm_flash_attn/_vllm_fa3_C.abi3.so` causes a ~10× throughput
> regression (measured H200 + Qwen3-8B ShareGPT: 15.6 → 1.5 RPS, TPOT
> 339 ms → 4158 ms). eb-vllm's diff vs upstream vLLM is **pure-Python**
> (`vllm/v1/core/sched/` + `reproduce/`), so the upstream precompiled FA3
> binary is ABI-compatible. Only build locally if you have modified `csrc/`,
> and in that case overwrite `vllm/vllm_flash_attn/_vllm_fa3_C.abi3.so`
> with the version extracted from the official wheel afterwards.

**Workload protocol** (paper §4.1 — match `IGNORE_EOS` + `CUSTOM_OUTPUT_LEN` per
workload):

| Workload | `CUSTOM_OUTPUT_LEN` | `IGNORE_EOS` |
|----------|---------------------|--------------|
| ShareGPT | `-1` (dataset-native, cap ≤500) | `true` |
| LongBench | `20` | `true` |
| NumimaMath | `4000` | `true` |
| WildChat | n/a (multi-turn chat) | n/a |

**EB⁺ runs (Table 4, Table 5)** additionally need the offline-profiled
mixed-batch cost coefficients (see Appendix `app:eb-plus-calibration`):
```bash
export VLLM_PD_MB_COST_A=2.494e-05   # H200 Qwen3-8B; recalibrate per (model, GPU)
export VLLM_PD_MB_COST_B=5.193e-05
export VLLM_PD_MB_COST_C=1.478e-05
export VLLM_PD_MODE_SWITCH_DELTA=1e-5
```

For end-to-end recipes with expected runtimes, see [`REPRODUCE.md`](REPRODUCE.md).

## Paper section → subdirectory

| Paper section | Artifact | Subdirectory |
|---|---|---|
| §3 Cost model — linear iteration-time | Fig. `execution_time*.pdf`, `prefill_linearity_all_models.png` | [`cost_model/linear_model/`](cost_model/linear_model/) |
| §3 Cost model — kernel breakdown | Fig. `kernel_breakdown*.pdf` | [`cost_model/kernel_breakdown/`](cost_model/kernel_breakdown/) |
| §3 Hazard rate / CFR vs IFR | Fig. `CFR_IFR.pdf`, `hazard_rate_comparison.pdf` | [`hazard_rate/`](hazard_rate/) |
| §4.2 Model validation | Fig. `validation_grid.pdf` | [`validation/`](validation/) |
| §4.3.1 Synthetic e2e | Fig. `fig_synthetic_e2e.pdf` | [`synthetic_e2e/`](synthetic_e2e/) |
| §4.3.2 Real workloads + §4.3.3 latency | Tables 2-3, Figs. `ttft.pdf`, `tpot.pdf` | [`real_workloads/`](real_workloads/) |
| §4.4 EB⁺ traffic-level | Table 4 | [`eb_plus/traffic/`](eb_plus/traffic/) |
| §4.4 EB⁺ non-stationary | Table 5 | [`eb_plus/non_stationary/`](eb_plus/non_stationary/) |
| §4.4 PD disaggregation comparison | Appendix | [`disagg/`](disagg/) |
| §4.4 Long-context comparison | Fig. `combined_ctx_comparison_tok1024.pdf` | [`long_context/`](long_context/) |
| §4.5.1-2 Scalability | Table 6, Fig. `scalmodel.pdf` | [`scalability/`](scalability/) |

## Shared infrastructure

- [`common/`](common/) — shared bash helpers (`common.sh`, `common_eb.sh`) and
  Python utilities (`dataset_utils.py`, `export_dataset.py`).
- [`calibration/`](calibration/) — cost-model calibration JSONs (one sample
  shipped; reviewers regenerate per (model, GPU) as needed).
- [`docker/`](docker/) — Dockerfile and build script for a reproducible
  environment.
- [`PD_SCHEDULER_ENV_VARS.md`](PD_SCHEDULER_ENV_VARS.md) — reference for all
  35 `VLLM_PD_*` env vars (mode selection, calibration, θ/N controllers,
  EB⁺ switching, diagnostics) with defaults, types, and audience tags.

## Recommended run order

See [`REPRODUCE.md`](REPRODUCE.md) for an annotated end-to-end recipe with
expected runtimes on a reference 8×RTX PRO 6000 machine.

## Notes

- Plot scripts marked **(demo)** use synthetic illustrative data when no real
  experiment output is present, so reviewers can preview the figure shape
  before running multi-hour experiments. Pass the real CSV path to render the
  paper version.
- Output files (`outputs/`, `__pycache__/`, etc.) are gitignored. Each
  `run_*.sh` writes results under its local `outputs/` subtree.
- `SKIP_EXISTING=1` (default) lets a re-run resume from where it stopped;
  a cell with an existing `bench_*.json` is skipped.

## Reproduction pitfalls (read this first if your numbers don't match paper)

Discovered while reproducing on RTX PRO 6000 (May 2026). Each item below was a
real blocker until fixed. **Items marked ✅ patched** have been fixed in this
branch; ⚠️ items remain in paper/setup level and require reader awareness.

### 1. ✅ Throughput unit standardized to total tokens/s (paper + code consistent)
Paper has been unified: §4 Metrics explicitly states "Throughput in tokens/s
(prefill+decode) and requests/s (RPS)" — all `tok/s` cells across Tab 4,
Tab 5, Disagg, e2e-128k now mean **total throughput** (input + output tokens).
vLLM's bench JSON `total_token_throughput` field matches paper.

Earlier versions of the paper had output-vs-total inconsistencies between
columns within the same table (verified via TTFT/TPOT self-consistency
check: `tput = concurrency / (TTFT + μ_O × TPOT) × tokens_per_req`); these
are now resolved.

### 2. ✅ Synthetic workloads use ±50% UNIFORM jitter, not CFR/geometric (patched)
Paper §4.1 spec: "Synthetic workloads fix mean input/output token counts
(with ±50% uniform jitter)". `run_grid_search.sh` and
`run_adaptive_selector.sh` previously hardcoded
`--dataset-name geometric_random` — now patched to default to `random` (uniform)
via env-override `DATASET_NAME=${DATASET_NAME:-random}`. To restore the old CFR
behavior explicitly:
```bash
DATASET_NAME=geometric_random bash run_grid_search.sh ...
```
Using `geometric_random` produces +5-15pp larger EB improvements on synthetic
than paper because the heavier tail of geometric output lengths favors
exclusive batching.

### 3. ✅ WildChat uses 3,000 conversations (NOT 500) — defaults patched
Paper `evaluation.tex` §4.1: "WildChat: 3,000 multi-turn, ~27,900 requests".
`multiturn/export_dataset.py` previously defaulted to 500 conv / min_turns=8;
now defaults to `--num-conversations 3000 --min-turns 6` (paper §4.1 setting):
```bash
python real_workloads/multiturn/export_dataset.py \
    --dataset wildchat --model Qwen/Qwen3-8B \
    --output reproduce/outputs/wildchat_multiturn.json
```
With 500 conv, server saturates fast → median TTFT under-reports paper by
10–25×. RPS comparable (server-bound), TTFT very different.

### 4. ✅ Scalability (Fig 7) uses Mathstral-7B-v0.1 (paper + code consistent)
Both paper §4.5.2 text and the figure plot script now use
`mistralai/Mathstral-7B-v0.1` consistently. The multiturn lookup table in
`real_workloads/multiturn/run_optimal_only.sh` includes Mathstral as one of
the 4 cross-model scalability entries. **Do not use Mistral-7B-v0.1** (base
model, no chat template — will fail with HTTP 400 on multi-turn).

### 5. ✅ PHASES env var collides with run_distribution_shift.sh (patched)
The one-click wrapper `run_all_paper_artifacts.sh` uses `PHASES` env var
to select phases (A/B/C/D/E). The inner script
`run_distribution_shift.sh` ALSO uses `PHASES` for its multi-phase dataset
spec ("1024:128,512:512,128:1024"). Without `unset PHASES`, the dataset
generator crashes ("not enough values to unpack"). Patched: wrapper now uses
`env -u PHASES` when invoking distshift/concshift children.

### 6. ✅ Calibration alpha_d can be 0 for some models (patched)
The calibrator fits decode latency as `T_d = α_d + β_d × k`. On a small
number of 7B base models we tested (none of them in the paper-reproduction
set), the regression yields `α_d = 0`. The scheduler then crashes on
`C = α_p / α_d` (division by zero, seen as `EngineDeadError` mid-benchmark).
Patched: `vllm/v1/core/sched/calibration.py` now floors `α_d` to
`max(1e-6, β_d * 0.01)` when regression returns 0, emitting a warning. If
the floored value looks unreasonable for your model, manually edit the JSON
with a proxy α_d from a similar model on the same GPU.

### 7. ✅ Missing Python dependencies (now in requirements/reproduce.txt)
The conda env created from `requirements/build.txt` does not include several
runtime deps used by reproduce scripts. Install with the new
`requirements/reproduce.txt`:
```bash
pip install -r requirements/reproduce.txt
```
Contains: `datasets` (HF datasets exports), `aiohttp` (multi-turn benchmark),
`quart` (disagg proxy), `matplotlib`/`pandas`/`scipy` (analyze + plot
scripts).

### 8. ✅ Stale `.venv` in repo root shadows conda env (patched)
Disaggregation scripts source `init_experiment_env` from `common/common.sh`,
which auto-sources `<repo>/.venv/bin/activate` if it exists. This previously
overrode an active conda env's Python path, causing `ModuleNotFoundError:
No module named 'vllm'`. Patched: `init_experiment_env` now checks
`$CONDA_PREFIX` and skips `.venv` activation if a conda env with `vllm`
in PATH is already active.

### 9. ✅ Missing (B, N) entries in `run_optimal_only.sh` lookup table (patched)
The (B, N) lookup case statement in `real_workloads/run_optimal_only.sh`
and `real_workloads/multiturn/run_optimal_only.sh` originally only covered
H200 × Qwen3-8B/30B-A3B + RTXPRO6000 × Qwen3-8B — RTXPRO6000 ×
Qwen3-30B-A3B + cross-model scalability entries were missing (script
exited with "no (B, N) entry"). Now patched with 12 RTX PRO 6000 ×
Qwen3-30B-A3B entries (from paper Appendix `tab:optimal-config-a6000`)
plus 4 cross-model wildchat entries (Llama-3.1-8B-Instruct,
Mathstral-7B-v0.1, Qwen2.5-Coder-7B, DeepSeek-R1-Distill-Qwen-7B).

### 10. ✅ `run_grid_search.sh` BS_VALUES/TB_VALUES env override (patched)
Lines 116-117 of `real_workloads/run_grid_search.sh` previously hardcoded
the BS/TB arrays. Patched to accept env-override:
```bash
BS_VALUES="1024 1536" TB_VALUES="8192 14336" \
    bash real_workloads/run_grid_search.sh ...
```

