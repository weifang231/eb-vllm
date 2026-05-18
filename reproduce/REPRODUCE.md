# Step-by-step reproduction recipe

Annotated end-to-end recipe for reproducing the EB(k̂*) paper (ICML 2026).
Times below are wall-clock estimates on 8×RTX PRO 6000 Blackwell (96 GB each).

## 0. Environment

```bash
# Build vLLM from this fork (release/icml2026 branch, tagged icml2026-camera-ready)
cd <repo-root>
python -m venv .venv && source .venv/bin/activate
pip install torch                                   # match your CUDA
VLLM_USE_PRECOMPILED=1 pip install -e . --no-build-isolation
                                # ~30 s; fetches all 8 .so (incl. FA3) from
                                # the official vLLM wheel. See FA3 warning
                                # below — do NOT skip this env var.
vllm --version                  # smoke check

# Required Python deps for analysis / plotting
pip install matplotlib numpy pandas scipy
```

A `docker/` recipe is provided for a fully reproducible environment;
the `vllm-openai` base image already ships the correct precompiled binaries.

> **⚠ FA3 warning — do NOT drop `VLLM_USE_PRECOMPILED=1`.**
> A locally-compiled `vllm/vllm_flash_attn/_vllm_fa3_C.abi3.so` causes a
> ~10× throughput regression. Measured on H200 + Qwen3-8B (ShareGPT):
>
> | configuration | RPS | TPOT |
> |---|---|---|
> | All 8 .so from official wheel (`VLLM_USE_PRECOMPILED=1`) | **15.65** | 339 ms |
> | Local-compiled FA3 only (other 7 .so from wheel)        | 1.48 | 4158 ms |
> | Fully fresh local install (`pip install -e .`)          | 1.47 | 4181 ms |
>
> Only FA3 matters — the other 7 .so can be either. eb-vllm's diff vs
> upstream vLLM is **pure-Python** (`vllm/v1/core/sched/` + `reproduce/`),
> so the upstream precompiled FA3 binary is ABI-compatible.
>
> If you must build C++/CUDA locally (e.g. you modified `csrc/`), overwrite
> `vllm/vllm_flash_attn/_vllm_fa3_C.abi3.so` with the version extracted
> from the official vLLM wheel after building. The other .so are safe to
> leave as-is.

## 1. Generate per-(model, GPU) calibration (one-time)

```bash
# Default --output auto-detects the GPU tag (H200 / RTXPRO6000 / ...) and
# writes to reproduce/calibration/pd_calibration_<model>_<GPU>.json — the
# same path the runner scripts (resolve_calibration in common_cfr.sh) look
# up. Pass --output explicitly only if you want a non-standard location.
python -m vllm.v1.core.sched.calibration --model Qwen/Qwen3-8B
```

Repeat for any other (model, GPU) you intend to run. The sample
`pd_calibration_Qwen3-8B_H200.json` is provided; the runner falls back
to a per-GPU file first, then the legacy non-GPU name.

## 2. §3 — Cost model validation (≈30 min)

```bash
cd cost_model/linear_model
python benchmark_execution_time.py --output results.json
python analyze_prefill_linearity_all.py
python plot_execution_time.py --output execution_time.pdf

cd ../kernel_breakdown
python benchmark_flash_attn_sweep.py --output results_$(hostname).json
python plot_kernel_breakdown.py --output kernel_breakdown.pdf
```

## 3. §3 — Hazard rate (≈1 h)

```bash
cd ../../hazard_rate
./run_hazard_rate_experiment.sh 4    # 4 GPUs
python analyze_hazard_rate.py outputs/
python plot_cfr_ifr.py --output CFR_IFR.pdf
python plot_sharegpt_hazard_rate.py --output hazard_rate_comparison.pdf
```

## 4. §4.2 — Controller validation (≈30 min)

```bash
cd ../validation
MODEL=Qwen/Qwen3-8B ./run_validation_cfr.sh 8
python analyze_cfr_validation.py outputs/controller_validation/<GPU>_Qwen3-8B
python plot_validation_grid.py outputs/.../validation_summary.csv \
    --output validation_grid.pdf
```

## 5. §4.3.1 — Synthetic e2e grid (≈1.5–8 h depending on model)

```bash
cd ../synthetic_e2e
MODEL=Qwen/Qwen3-8B    ./run_grid_search_cfr.sh 8    # ~1.5 h
MODEL=Qwen/Qwen3-30B-A3B ./run_grid_search_cfr.sh 8  # ~7-8 h
python analyze_cfr_e2e.py outputs/e2e_grid_search/<GPU>_<MODEL>
python plot_synthetic_e2e.py --output fig_synthetic_e2e.pdf
```

## 6. §4.3.2/3 — Real workloads (≈6–10 h per (model, GPU))

```bash
cd ../real_workloads
MODEL=Qwen/Qwen3-8B GPUS=0,1,2,3 ./run_grid_search.sh
python analyze_grid_search.py outputs/<GPU>_<MODEL>/
python plot_real_workload_latency.py \
    --optimal-json outputs/<GPU>_<MODEL>/optimal_per_scheduler.json \
    --ttft-output ttft.pdf --tpot-output tpot.pdf
```

**Workload protocol** (paper §4.1; set `IGNORE_EOS` and `CUSTOM_OUTPUT_LEN`
to match the per-workload truncation cap):

| Workload | `CUSTOM_OUTPUT_LEN` | `IGNORE_EOS` | Notes |
|----------|---------------------|--------------|-------|
| ShareGPT | `-1` (dataset-native, cap ≤500) | `true` | output mean ≈ 280 tok |
| LongBench | `20` | `true` | forced short output |
| NumimaMath | `4000` | `true` | forced full chain-of-thought |
| WildChat | n/a | n/a | multi-turn chat-mode, see `multiturn/` |

Single-cell example (optimal `(B, N)` from Appendix `tab:optimal-config-h200`):
```bash
IGNORE_EOS=true CUSTOM_OUTPUT_LEN=500 MODEL=Qwen/Qwen3-8B \
    ./run_optimal_only.sh ../outputs/sharegpt_prompts.jsonl 1
IGNORE_EOS=true CUSTOM_OUTPUT_LEN=20 MODEL=Qwen/Qwen3-8B \
    ./run_optimal_only.sh ../outputs/longbench_prefill.jsonl 1
IGNORE_EOS=true CUSTOM_OUTPUT_LEN=4000 MODEL=Qwen/Qwen3-8B \
    ./run_optimal_only.sh ../outputs/numina_math_prompts.jsonl 1
```

For multi-turn dialogue (WildChat) preprocessing, see
[`real_workloads/multiturn/`](real_workloads/multiturn/).

## 7. §4.4 — EB⁺ (≈30 min for traffic, ≈2 h for non-stationary)

**EB⁺ requires offline mixed-batch cost calibration** (Appendix
`app:eb-plus-calibration`). The default coefficients for H200 Qwen3-8B
are shipped in `reproduce/calibration/beta_mb_Qwen3-8B_H200.json`. For
other (model, GPU) pairs, refit from v1 grid data and update env vars:

```bash
# H200 Qwen3-8B (provided)
export VLLM_PD_CP_COST_A=2.494e-05
export VLLM_PD_CP_COST_B=5.193e-05
export VLLM_PD_CP_COST_C=1.478e-05
# Mode-switch hysteresis δ — match observed |LHS-RHS| magnitude
export VLLM_PD_MODE_SWITCH_DELTA=1e-5
```

Then run the EB⁺ experiments:
```bash
cd ../eb_plus/traffic
MODEL=Qwen/Qwen3-8B ./run_adaptive_selector_cfr.sh 8
python analyze_cfr_selector.py outputs/adaptive_selector/<GPU>_<MODEL>

cd ../non_stationary
python generate_distribution_shift_dataset.py
./run_distribution_shift.sh 0    # positional arg = GPU_ID, not MAX_GPUS
./run_concurrency_shift.sh 1
python plot_distribution_shift.py outputs/
```

If `pd_auto_stats.json` shows `mode_switch_count = 0`, the crossover
hysteresis `VLLM_PD_MODE_SWITCH_DELTA` is too large for the observed
LHS-RHS magnitude — try `1e-6` (or refit `(a, b, c)`).

## 8. §4.4 — Disaggregation comparison (≈4 h on 4 GPUs)

```bash
cd ../../disagg
./run_2gpu_comparison.sh
./run_4gpu_comparison.sh
./run_disagg_baseline.sh
```

## 9. §4.4 — Long context (≈2 h)

```bash
cd ../long_context
./run_long_context_comparison.sh
./run_128k_all_models.sh
python plot_long_context.py outputs/long_context_summary.csv \
    --output combined_ctx_comparison_tok1024.pdf
```

## 10. §4.5 — Scalability (reuses real_workloads, swap GPU/MODEL)

See [`scalability/README.md`](scalability/) for the GPU/model
matrix and the `plot_scalmodel.py` invocation.

## Total time

End-to-end full reproduction is **roughly one week** on 8×RTX 6000.
For a sampler tour, run §3 + §4.2 + §4.3.1 synthetic (≈4 h total).
