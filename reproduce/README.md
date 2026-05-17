# Reproducing the THETA paper (ICML 2026)

This directory contains the experiment scripts, analysis tools, and plot
scripts that reproduce every figure and table in the paper. Each
`secX_Y_*/` subdirectory corresponds to a specific paper section.

## Quick Start

```bash
# 1. Clone + setup environment
git clone https://github.com/weifang231/eb-vllm
cd eb-vllm
git checkout release/icml2026
pip install -e .                # vLLM editable install (~20 min compile)
pip install matplotlib numpy pandas scipy

# 2. One-time per-(model, GPU) cost-model calibration
python -m vllm.v1.core.sched.calibration \
    --model Qwen/Qwen3-8B \
    --output reproduce/calibration/pd_calibration_Qwen3-8B_<GPU_TAG>.json
# (Sample H200 / RTX PRO 6000 calibrations shipped under reproduce/calibration/.)

# 3. Run one paper artifact (example: Table 3 H200, Qwen3-8B ShareGPT)
cd reproduce/real_workloads
IGNORE_EOS=true CUSTOM_OUTPUT_LEN=500 MODEL=Qwen/Qwen3-8B \
    bash run_optimal_only.sh ../outputs/sharegpt_prompts.jsonl 1

# 4. Analyze / plot
python analyze_grid_search.py ../outputs/optimal_only_sharegpt_prompts_Qwen3-8B_*
```

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
export VLLM_PD_CP_COST_A=2.494e-05   # H200 Qwen3-8B; recalibrate per (model, GPU)
export VLLM_PD_CP_COST_B=5.193e-05
export VLLM_PD_CP_COST_C=1.478e-05
export VLLM_PD_MODE_SWITCH_DELTA=1e-5
```

For end-to-end recipes with expected runtimes, see [`REPRODUCE.md`](REPRODUCE.md).
For per-figure reproduction status, see [`REPRODUCTION_REPORT.md`](REPRODUCTION_REPORT.md).

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

- [`common/`](common/) — shared bash helpers (`common.sh`, `common_cfr.sh`) and
  Python utilities (`dataset_utils.py`, `export_dataset.py`).
- [`calibration/`](calibration/) — cost-model calibration JSONs (one sample
  shipped; reviewers regenerate per (model, GPU) as needed).
- [`docker/`](docker/) — Dockerfile and build script for a reproducible
  environment.

## Recommended run order

See [`REPRODUCE.md`](REPRODUCE.md) for an annotated end-to-end recipe with
expected runtimes on a reference 8×RTX PRO 6000 machine.

## Reproduction status

See [`REPRODUCTION_REPORT.md`](REPRODUCTION_REPORT.md) for per-figure
reproducibility status (✅ / ⚠️ / ⛔), exact commands, and notes on how the
implementation maps to the analytical statements in the paper.

## Notes

- Plot scripts marked **(demo)** use synthetic illustrative data when no real
  experiment output is present, so reviewers can preview the figure shape
  before running multi-hour experiments. Pass the real CSV path to render the
  paper version.
- Output files (`outputs/`, `__pycache__/`, etc.) are gitignored. Each
  `run_*.sh` writes results under its local `outputs/` subtree.
- `SKIP_EXISTING=1` (default) lets a re-run resume from where it stopped;
  a cell with an existing `bench_*.json` is skipped.
